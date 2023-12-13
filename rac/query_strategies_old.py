import numpy as np 
from itertools import combinations, product
from scipy.special import softmax

class QueryStrategy:
    def __init__(self, ac):
        self.ac = ac
        self.custom_informativeness = None

    def select_batch(self, acq_fn, local_regions, batch_size):

        if "info_gain" in acq_fn:
            self.custom_informativeness = self.compute_information_gain_matrix(batch_size)
            return np.array(self.select_pairs("custom", "all_pairs", batch_size)), None

        if "maxmin" in acq_fn or "maxexp" in acq_fn:
            if "2" in acq_fn:
                self.compute_informativeness_2(acq_fn)
            else:
                self.compute_informativeness(acq_fn)

        if type(local_regions) == str:
            if local_regions == "pairs":
                return np.array(self.select_pairs(acq_fn, "all_pairs", batch_size)), None
            elif local_regions == "triangles":
                if self.ac.random.rand() < self.ac.eps:
                    return np.array(self.select_pairs("freq", "all_pairs", batch_size)), None
                else:
                    #self.compute_informativeness(acq_fn)
                    return np.array(self.select_pairs("custom", "all_pairs", batch_size)), None
            elif local_regions == "clusters":
                local_regions = self.clusters()
            else:
                raise ValueError("Invalid local region type. Must be one of 'pairs', 'triangles', or 'clusters'.")

        acq_fn = getattr(self, acq_fn)
        sorted_regions = self.sort_local_regions(local_regions, acq_fn)
        selected_edges = []
        for lr_ind in sorted_regions:
            local_region = local_regions[lr_ind]
            if len(selected_edges) == batch_size:
                break
            num_selections = min(batch_size - len(selected_edges), len(local_region))
            selected_edges.extend(self.select_pairs("uncert", local_region, num_selections))
        return np.array(selected_edges), None

    def unif(self, local_region):
        return self.ac.random.rand(1)[0]

    def freq(self, local_region):
        return self.index_matrix(local_region, -self.ac.feedback_freq).sum() / len(local_region)

    def uncert(self, local_region):
        return self.index_matrix(local_region, -np.abs(self.ac.pairwise_similarities)).sum() / len(local_region)

    def incon(self, local_region):
        return self.index_matrix(local_region, self.ac.violations).sum() / len(local_region)

    def incon_ucb(self, local_region):
        return self.incon(local_region) + self.ac.alpha * self.uncert(local_region)

    def maxmin(self, local_region):
        return self.index_matrix(local_region, self.custom_informativeness).sum() / len(local_region)

    def maxexp(self, local_region):
        return self.index_matrix(local_region, self.custom_informativeness).sum() / len(local_region)

    def maxmin_ucb(self, local_region):
        return self.maxmin(local_region) + self.ac.alpha * self.uncert(local_region)

    def maxexp_ucb(self, local_region):
        return self.maxexp(local_region) + self.ac.alpha * self.uncert(local_region)

    def triangles(self):
        pass

    def compute_cluster_probs(self):
        # Number of objects (N) and clusters (M)
        N = self.ac.pairwise_similarities.shape[0]
        M = len(self.ac.clustering)

        # Initialize the NxM matrix to hold summed similarities of each object to each cluster
        similarity_matrix = np.zeros((N, M))

        # For each cluster, sum the similarities for each object
        for cluster_index, cluster in enumerate(self.ac.clustering):
            # Sum the similarities of each object to all objects in the cluster
            # Note that we are using NumPy broadcasting here.
            similarity_matrix[:, cluster_index] = self.ac.pairwise_similarities[:, cluster].sum(axis=1)

        # Now compute the probability matrix by dividing each summed similarity by the sum across all clusters
        prob_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)

        return prob_matrix

    def compute_conditional_entropy(self, prob_matrix, object_i, object_j, same_cluster):
        N, M = prob_matrix.shape
        # For simplicity, create a deep copy to not modify the original matrix
        conditional_prob_matrix = np.copy(prob_matrix)
        
        if same_cluster:
            # Calculate the joint probability for each cluster
            for k in range(M):
                joint_prob = prob_matrix[object_i, k] * prob_matrix[object_j, k]
                conditional_prob_matrix[object_i, k] = joint_prob
                conditional_prob_matrix[object_j, k] = joint_prob
        else:
            # Calculate the probabilities considering they are in different clusters
            for k in range(M):
                conditional_prob_matrix[object_i, k] *= (1 - prob_matrix[object_j, k])
                conditional_prob_matrix[object_j, k] *= (1 - prob_matrix[object_i, k])

        # Normalize the updated probabilities for each object
        conditional_prob_matrix[object_i, :] /= np.sum(conditional_prob_matrix[object_i, :])
        conditional_prob_matrix[object_j, :] /= np.sum(conditional_prob_matrix[object_j, :])

        conditional_prob_matrix = np.clip(conditional_prob_matrix, 1e-10, 1)

        # Compute the entropy for the conditional matrix
        H_C_given_R = 0
        for i in range(N):
            H_i = -np.sum(conditional_prob_matrix[i, :] * np.log(conditional_prob_matrix[i, :]))  # Added a small constant to prevent log(0)
            H_C_given_R += H_i

        return H_C_given_R

    def compute_expected_conditional_entropy(self, prob_matrix, object_i, object_j):
        M = prob_matrix.shape[1]
        
        # Compute the probability that objects i and j are in the same cluster
        P_same = sum(prob_matrix[object_i, k] * prob_matrix[object_j, k] for k in range(M))
        
        # Compute the probability that objects i and j are in different clusters
        P_diff = 1 - P_same
        
        # Compute the conditional entropy given that i and j are in the same cluster
        H_C_given_same = self.compute_conditional_entropy(prob_matrix, object_i, object_j, True)
        
        # Compute the conditional entropy given that i and j are in different clusters
        H_C_given_diff = self.compute_conditional_entropy(prob_matrix, object_i, object_j, False)
        
        # Compute the expected conditional entropy
        H_C_given_R = P_same * H_C_given_same + P_diff * H_C_given_diff
        
        return H_C_given_R
 
    def compute_information_gain_matrix(self, batch_size):
        prob_matrix = self.compute_cluster_probs()
        N = prob_matrix.shape[0]

        # Initialize entropy of clustering
        H_C = 0

        # Iterate over each object
        prob_matrix = np.clip(prob_matrix, 1e-10, 1)
        for i in range(N):
            # For each object, compute the contribution to the entropy from each cluster assignment
            H_i = -np.sum(prob_matrix[i, :] * np.log(prob_matrix[i, :]))  # Added a small constant to prevent log(0)
            H_C += H_i

        N = prob_matrix.shape[0]
        info_gain_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):  # No need to compute for j <= i because the matrix is symmetric
                # Compute the expected conditional entropy H(C | R) for each pair (i, j)
                H_C_given_R = self.compute_expected_conditional_entropy(prob_matrix, i, j)
                
                # Compute the information gain for pair (i, j)
                info_gain = H_C - H_C_given_R
                
                # Fill in the symmetric matrix
                info_gain_matrix[i, j] = info_gain
                info_gain_matrix[j, i] = info_gain
                
        return info_gain_matrix
        
    
    def compute_info_gain(self, h, q, S):
        if self.ac.clustering_alg != "mean_field":
            raise ValueError("Info gain is only defined for mean field clustering")

        N, K = q.shape
        I = np.zeros((N, N))

        beta = self.ac.mean_field_beta
        #beta = self.ac.info_gain_beta
        lmbda = self.ac.info_gain_lambda
        lmbda2 = self.ac.info_gain_lambda2
        num_edges = self.ac.num_edges_info_gain if self.ac.num_edges_info_gain > 0 else N

        q = scipy_softmax(beta*-h, axis=1)
        h = -np.dot(S, q)

        H_C = np.sum(scipy_entropy(q, axis=1))
        #print("HAASDSAD: ", H_C)
        n_iter = 20

        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(self.ac.feedback_freq[lower_triangle_indices] > 0)[0]
        num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else len(inds)
        inds = np.random.choice(inds, num_edges, replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]

        #print("ASDSD: ", len(a))
        for x, y in zip(a, b):
        #for x in range(N):
            #for y in range(x):
            S_xy = S[x, y]

            # Compute h for P(C | e = 1)
            h_e1 = np.copy(h)
            q_e1 = np.copy(q)

            h_e1[x, :] += q_e1[y, :] * lmbda2 * (S_xy - lmbda)
            h_e1[y, :] += q_e1[x, :] * lmbda2 * (S_xy - lmbda)
            q_e1[x, :] = scipy_softmax(beta * -h_e1[x, :])
            q_e1[y, :] = scipy_softmax(beta * -h_e1[y, :])

            if self.ac.iterate_mf: 
                for i in range(n_iter):
                    h_e1 = -np.dot(S, q_e1)
                    h_e1[x, :] += q_e1[y, :] * lmbda2 * (S_xy - lmbda)
                    h_e1[y, :] += q_e1[x, :] * lmbda2 * (S_xy - lmbda)
                    q_e1 = scipy_softmax(beta*-h_e1, axis=1)
            H_C_e1 = np.sum(scipy_entropy(q_e1, axis=1))

            # Compute h for P(C | e = 1)
            h_em1 = np.copy(h)
            q_em1 = np.copy(q)

            h_em1[x, :] += q_em1[y, :] * lmbda2 * (S_xy + lmbda)
            h_em1[y, :] += q_em1[x, :] * lmbda2 * (S_xy + lmbda)
            q_em1[x, :] = scipy_softmax(beta * -h_em1[x, :])
            q_em1[y, :] = scipy_softmax(beta * -h_em1[y, :])

            if self.ac.iterate_mf: 
                for i in range(n_iter):
                    h_em1 = -np.dot(S, q_em1)
                    h_em1[x, :] += q_em1[y, :] * lmbda2 * (S_xy + lmbda)
                    h_em1[y, :] += q_em1[x, :] * lmbda2 * (S_xy + lmbda)
                    q_em1 = scipy_softmax(beta*-h_em1, axis=1)
            H_C_em1 = np.sum(scipy_entropy(q_em1, axis=1))


            P_e1 = np.sum(q[x, :] * q[y, :])
            P_e_minus_1 = 1 - P_e1 
            
            H_C_e = P_e1 * H_C_e1 + P_e_minus_1 * H_C_em1
            #print("H_C_e: ", H_C_e)
            I[x, y] = H_C - H_C_e
            I[y, x] = I[x, y]
            #print("H_C: ", H_C)
            #print("H_C_e: ", H_C_e)
            #print("P_e1: ", P_e1)
            #print("P_e_minus_1: ", P_e_minus_1)
            #print("H_C_e1: ", H_C_e1)
            #print("H_C_e_minus_1: ", H_C_e_minus_1)
            #print("I: ", I[x, y])
        #self.sort_similarity_matrix(I)
        #self.sort_similarity_matrix(self.ac.pairwise_similarities)
        return I


    def clusters(self):
        local_regions = []
        for i, cluster_i in enumerate(self.ac.clustering):
            for j, cluster_j in enumerate(self.ac.clustering):
                if i != j:
                    pairwise_indices = list(product(cluster_i, cluster_j))
                    local_regions.append(pairwise_indices)
                else:
                    if len(cluster_i) == 1:
                        continue
                    pairwise_indices = list(combinations(cluster_i, 2))
                    local_regions.append(pairwise_indices)
        return local_regions

    def all_pairs(self):
        indices = np.arange(self.ac.N)
        row_indices, col_indices = np.tril_indices(self.ac.N, k=-1)
        return [(indices[row], indices[col]) for row, col in zip(row_indices, col_indices)]

    def sort_local_regions(self, local_regions, acq_fn):
        acq_vals = np.array([acq_fn(local_region) for local_region in local_regions])
        noise = np.random.uniform(-1e-10, 1e-10, acq_vals.shape)
        acq_vals += noise
        sorted_indices = np.argsort(-acq_vals)
        return sorted_indices

    def index_matrix(self, local_region, matrix):
        row_indices, col_indices = zip(*local_region)
        return matrix[row_indices, col_indices] 

    def select_pairs(self, acq_fn, edges, batch_size):
        if acq_fn == "unif":
            S = self.ac.random.rand(self.ac.N, self.ac.N)
        elif acq_fn == "freq":
            S = -self.ac.feedback_freq
        elif acq_fn == "uncert":
            S = -np.abs(self.ac.pairwise_similarities)
        elif acq_fn == "custom":
            S = self.custom_informativeness
        else:
            raise ValueError("Invalid acquisition function. Must be one of 'unif', 'freq', or 'uncert'.")

        if edges == "all_pairs":
            edges = self.all_pairs()

        subset_flat = S.flatten()
        pairs = np.array(edges)

        values = subset_flat[np.ravel_multi_index((pairs[:, 0], pairs[:, 1]), S.shape)]
        values += np.random.rand(len(values)) * 1e-10  # Add small random noise to break ties (if any

        top_M_indices_unsorted = np.argpartition(values, -batch_size)[-batch_size:]
        top_M_indices = top_M_indices_unsorted[np.argsort(-values[top_M_indices_unsorted])]

        top_M_values = values[top_M_indices]
        pairwise_indices = pairs[top_M_indices]

        #result = [(tuple(indices), value) for indices, value in zip(pairwise_indices, top_M_values)]
        result = [tuple(indices) for indices, value in zip(pairwise_indices, top_M_values)]

        return result

    def min_triple_cost(self, i, j, k, beta=1):
        sim_ij = self.ac.pairwise_similarities[i, j]
        sim_ik = self.ac.pairwise_similarities[i, k]
        sim_jk = self.ac.pairwise_similarities[j, k]

        # (i, j, k)
        c1 = 0
        if sim_ij < 0:
            c1 += np.abs(sim_ij)

        if sim_ik < 0:
            c1 += np.abs(sim_ik)

        if sim_jk < 0:
            c1 += np.abs(sim_jk)

        # (i, k), (j)
        c2 = 0
        if sim_ij >= 0:
            c2 += np.abs(sim_ij)

        if sim_ik < 0:
            c2 += np.abs(sim_ik)

        if sim_jk >= 0:
            c2 += np.abs(sim_jk)

        # (i, j), (k)
        c3 = 0
        if sim_ij < 0:
            c3 += np.abs(sim_ij)

        if sim_ik >= 0:
            c3 += np.abs(sim_ik)

        if sim_jk >= 0:
            c3 += np.abs(sim_jk)

        # (k, j), (i)
        c4 = 0
        if sim_ij >= 0:
            c4 += np.abs(sim_ij)

        if sim_ik >= 0:
            c4 += np.abs(sim_ik)

        if sim_jk < 0:
            c4 += np.abs(sim_jk)

        # (i), (j), (k)
        c5 = 0
        if sim_ij >= 0:
            c5 += np.abs(sim_ij)

        if sim_ik >= 0:
            c5 += np.abs(sim_ik)

        if sim_jk >= 0:
            c5 += np.abs(sim_jk)

        costs = np.array([c1, c2, c3, c4, c5])
        probs = softmax(-beta*costs)
        expected_loss = np.sum(costs*probs)

        return expected_loss

    def triangle_is_bad(self, i, j, k):
        sim_ij = self.ac.pairwise_similarities[i, j]
        sim_ik = self.ac.pairwise_similarities[i, k]
        sim_jk = self.ac.pairwise_similarities[j, k]
        num_pos = 0

        if sim_ij > 0:
            num_pos +=1

        if sim_ik > 0:
            num_pos +=1

        if sim_jk > 0:
            num_pos +=1

        return num_pos == 2

    def in_same_cluster(self, o1, o2):
        return self.ac.clustering_solution[o1] == self.ac.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] < 0)
    
    def random_sort(self, arr, ascending=True):
        if not ascending:
            arr = -arr
        b = self.ac.random.random(arr.size)
        return np.lexsort((b, arr))

    #def compute_informativeness(self, acq_fn):
    #    if acq_fn == "maxmin":
    #        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
    #        N = self.ac.N
    #        for (i, j) in self.ac.queried_edges:
    #            for k in range(0, N):
    #                if k == i or k == j:
    #                    continue
    #                sim_ij = self.ac.pairwise_similarities[i, j]
    #                sim_ik = self.ac.pairwise_similarities[i, k]
    #                sim_jk = self.ac.pairwise_similarities[j, k]

    #                if self.triangle_is_bad(i, j, k):
    #                    sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
    #                    smallest_sim = self.random_sort(np.array(sims))[0]

    #                    if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
    #                        self.custom_informativeness[i, j] = np.abs(sim_ij)
    #                        self.custom_informativeness[j, i] = np.abs(sim_ij)
    #                    if smallest_sim == 1 and self.ac.feedback_freq[i, k] <= self.ac.tau:
    #                        self.custom_informativeness[i, k] = np.abs(sim_ik)
    #                        self.custom_informativeness[k, i] = np.abs(sim_ik)
    #                    if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
    #                        self.custom_informativeness[j, k] = np.abs(sim_jk)
    #                        self.custom_informativeness[k, j] = np.abs(sim_jk)
    #    elif acq_fn == "maxexp":
    #        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
    #        N = self.ac.N
    #        for (i, j) in self.ac.queried_edges:
    #            for k in range(0, N):
    #                if k == i or k == j:
    #                    continue
    #                sim_ij = self.ac.pairwise_similarities[i, j]
    #                sim_ik = self.ac.pairwise_similarities[i, k]
    #                sim_jk = self.ac.pairwise_similarities[j, k]

    #                if self.triangle_is_bad(i, j, k):# and (self.ac.feedback_freq[i, j] > 1 or self.ac.feedback_freq[i, k] > 1 or self.ac.feedback_freq[j, k] > 1):
    #                    expected_loss = self.min_triple_cost(i, j, k, beta=self.ac.beta) 
    #                    sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
    #                    smallest_sim = self.random_sort(np.array(sims))[0]
    #                    if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
    #                        self.custom_informativeness[i, j] = np.maximum(expected_loss, self.custom_informativeness[i, j])
    #                        self.custom_informativeness[j, i] = np.maximum(expected_loss, self.custom_informativeness[j, i])
    #                    if smallest_sim == 1 and self.ac.feedback_freq[i, k] <=  self.ac.tau:
    #                        self.custom_informativeness[i, k] = np.maximum(expected_loss, self.custom_informativeness[i, k])
    #                        self.custom_informativeness[k, i] = np.maximum(expected_loss, self.custom_informativeness[k, i])
    #                    if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
    #                        self.custom_informativeness[j, k] = np.maximum(expected_loss, self.custom_informativeness[j, k])
    #                        self.custom_informativeness[k, j] = np.maximum(expected_loss, self.custom_informativeness[k, j])
    #    else:
    #        raise ValueError("Invalid acquisition function maxmin/maxexp.")

    def compute_informativeness(self, acq_fn):
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
        if acq_fn == "maxmin":
            self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
            N = self.ac.N
            for i, j in zip(a, b):
                if self.violates_clustering(i, j):# and self.ac.feedback_freq[i, j] > 1: # adding this condition should not change anything when there are no bad triangles initially! since all new triangles must include at least one queried edge
                    for k in range(0, N):
                        if k == i or k == j:
                            continue
                        sim_ij = self.ac.pairwise_similarities[i, j]
                        sim_ik = self.ac.pairwise_similarities[i, k]
                        sim_jk = self.ac.pairwise_similarities[j, k]

                        if self.triangle_is_bad(i, j, k):# and (self.ac.feedback_freq[i, j] > 1 or self.ac.feedback_freq[i, k] > 1 or self.ac.feedback_freq[j, k] > 1):
                            sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                            smallest_sim = self.random_sort(np.array(sims))[0]

                            if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
                                self.custom_informativeness[i, j] = np.abs(sim_ij)
                                self.custom_informativeness[j, i] = np.abs(sim_ij)
                            if smallest_sim == 1 and self.ac.feedback_freq[i, k] <= self.ac.tau:
                                self.custom_informativeness[i, k] = np.abs(sim_ik)
                                self.custom_informativeness[k, i] = np.abs(sim_ik)
                            if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
                                self.custom_informativeness[j, k] = np.abs(sim_jk)
                                self.custom_informativeness[k, j] = np.abs(sim_jk)
        elif acq_fn == "maxexp":
            self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
            N = self.ac.N
            for i, j in zip(a, b):
                if self.violates_clustering(i, j):# and self.ac.feedback_freq[i, j] > 1:
                    for k in range(0, N):
                        if k == i or k == j:
                            continue
                        sim_ij = self.ac.pairwise_similarities[i, j]
                        sim_ik = self.ac.pairwise_similarities[i, k]
                        sim_jk = self.ac.pairwise_similarities[j, k]

                        if self.triangle_is_bad(i, j, k):# and (self.ac.feedback_freq[i, j] > 1 or self.ac.feedback_freq[i, k] > 1 or self.ac.feedback_freq[j, k] > 1):
                            expected_loss = self.min_triple_cost(i, j, k, beta=self.ac.beta) 
                            sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                            smallest_sim = self.random_sort(np.array(sims))[0]
                            if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
                                self.custom_informativeness[i, j] = np.maximum(expected_loss, self.custom_informativeness[i, j])
                                self.custom_informativeness[j, i] = np.maximum(expected_loss, self.custom_informativeness[j, i])
                            if smallest_sim == 1 and self.ac.feedback_freq[i, k] <=  self.ac.tau:
                                self.custom_informativeness[i, k] = np.maximum(expected_loss, self.custom_informativeness[i, k])
                                self.custom_informativeness[k, i] = np.maximum(expected_loss, self.custom_informativeness[k, i])
                            if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
                                self.custom_informativeness[j, k] = np.maximum(expected_loss, self.custom_informativeness[j, k])
                                self.custom_informativeness[k, j] = np.maximum(expected_loss, self.custom_informativeness[k, j])
        else:
            raise ValueError("Invalid acquisition function maxmin/maxexp.")

    def compute_informativeness_2(self, acq_fn):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        max_edges = self.ac.N
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        inds = self.ac.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]

        if acq_fn == "maxmin2":
            self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
            N = self.ac.N
            for i, j in zip(a, b):
                if self.violates_clustering(i, j):# and self.ac.feedback_freq[i, j] > 1: # adding this condition should not change anything when there are no bad triangles initially! since all new triangles must include at least one queried edge
                    for k in range(0, N):
                        if k == i or k == j:
                            continue
                        sim_ij = self.ac.pairwise_similarities[i, j]
                        sim_ik = self.ac.pairwise_similarities[i, k]
                        sim_jk = self.ac.pairwise_similarities[j, k]

                        if self.triangle_is_bad(i, j, k):# and (self.ac.feedback_freq[i, j] > 1 or self.ac.feedback_freq[i, k] > 1 or self.ac.feedback_freq[j, k] > 1):
                            sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                            smallest_sim = self.random_sort(np.array(sims))[0]

                            if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
                                self.custom_informativeness[i, j] = np.abs(sim_ij)
                                self.custom_informativeness[j, i] = np.abs(sim_ij)
                            if smallest_sim == 1 and self.ac.feedback_freq[i, k] <= self.ac.tau:
                                self.custom_informativeness[i, k] = np.abs(sim_ik)
                                self.custom_informativeness[k, i] = np.abs(sim_ik)
                            if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
                                self.custom_informativeness[j, k] = np.abs(sim_jk)
                                self.custom_informativeness[k, j] = np.abs(sim_jk)
        elif acq_fn == "maxexp2":
            self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
            N = self.ac.N
            for i, j in zip(a, b):
                if self.violates_clustering(i, j):# and self.ac.feedback_freq[i, j] > 1:
                    for k in range(0, N):
                        if k == i or k == j:
                            continue
                        sim_ij = self.ac.pairwise_similarities[i, j]
                        sim_ik = self.ac.pairwise_similarities[i, k]
                        sim_jk = self.ac.pairwise_similarities[j, k]

                        if self.triangle_is_bad(i, j, k):# and (self.ac.feedback_freq[i, j] > 1 or self.ac.feedback_freq[i, k] > 1 or self.ac.feedback_freq[j, k] > 1):
                            expected_loss = self.min_triple_cost(i, j, k, beta=self.ac.beta) 
                            sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                            smallest_sim = self.random_sort(np.array(sims))[0]
                            if smallest_sim == 0 and self.ac.feedback_freq[i, j] <= self.ac.tau:
                                self.custom_informativeness[i, j] = np.maximum(expected_loss, self.custom_informativeness[i, j])
                                self.custom_informativeness[j, i] = np.maximum(expected_loss, self.custom_informativeness[j, i])
                            if smallest_sim == 1 and self.ac.feedback_freq[i, k] <=  self.ac.tau:
                                self.custom_informativeness[i, k] = np.maximum(expected_loss, self.custom_informativeness[i, k])
                                self.custom_informativeness[k, i] = np.maximum(expected_loss, self.custom_informativeness[k, i])
                            if smallest_sim == 2 and self.ac.feedback_freq[j, k] <= self.ac.tau:
                                self.custom_informativeness[j, k] = np.maximum(expected_loss, self.custom_informativeness[j, k])
                                self.custom_informativeness[k, j] = np.maximum(expected_loss, self.custom_informativeness[k, j])
        else:
            raise ValueError("Invalid acquisition function maxmin/maxexp.")

    