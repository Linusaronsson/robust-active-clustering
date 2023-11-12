import numpy as np 
from itertools import combinations, product
from scipy.special import softmax

class QueryStrategy:
    def __init__(self, ac):
        self.ac = ac
        self.info_matrix = None

    def select_batch(self, acq_fn, batch_size):
        if acq_fn == "unif":
            self.info_matrix = np.random.rand(self.ac.N, self.ac.N)
        elif acq_fn == "freq":
            self.info_matrix = -self.ac.feedback_freq
        elif acq_fn == "uncert":
            self.info_matrix = -np.abs(self.ac.pairwise_similarities)
        elif acq_fn == "incon":
            self.info_matrix = self.ac.violations
        elif acq_fn == "maxmin":
            if np.random.rand() < self.ac.eps:
                self.info_matrix = -self.ac.feedback_freq
            else:
                self.info_matrix = self.compute_maxmin()
        elif acq_fn == "maxexp":
            if np.random.rand() < self.ac.eps:
                self.info_matrix = -self.ac.feedback_freq
            else:
                self.info_matrix = self.compute_maxexp()
        elif acq_fn == "info_gain":
            self.info_matrix = self.compute_info_gain(self.ac.h, self.ac.q, self.ac.pairwise_similarities)
            pass
        else:
            raise ValueError("Invalid acquisition function: {}".format(acq_fn))

        return self.select_edges(batch_size)
           

    def select_edges(self, batch_size):
        I = self.info_matrix

        inds_max_query = np.where(self.ac.feedback_freq > self.ac.tau)
        I[inds_max_query] = -np.inf

        # Get the lower triangular indices, excluding the diagonal
        tri_rows, tri_cols = np.tril_indices(n=I.shape[0], k=-1)
        
        # Get the informativeness scores for the lower triangular part
        informative_scores = I[tri_rows, tri_cols]
        
        # Generate a random array to break ties randomly
        random_tiebreaker = np.random.rand(informative_scores.shape[0])
        
        # Use lexsort to sort by informativeness first, then by the random tiebreaker
        sorted_indices = np.lexsort((random_tiebreaker, -informative_scores))
        
        # Select the top B indices
        top_indices = sorted_indices[:batch_size]
        
        # Get the corresponding row and column indices
        top_row_indices = tri_rows[top_indices]
        top_col_indices = tri_cols[top_indices]
        
        # Stack the row and column indices to get the desired output format
        top_pairs = np.stack((top_row_indices, top_col_indices), axis=-1)
        
        # Return the top pairs in descending order of their informativeness
        return top_pairs
        
    def select_edges2(self, I, B):
        # Ensure the input is a numpy array
        I = self.info_matrix

        # Add small noise to the entire matrix to handle ties randomly
        noise = np.random.uniform(0, 1e-12, I.shape)
        I_noise = I + noise
        
        # Get the indices of the lower triangular part of the matrix, excluding the diagonal
        lower_tri_indices = np.tril_indices_from(I, k=-1)
        
        # Flatten the informative matrix and take only the lower triangular part, excluding the diagonal
        flat_I_noise = I_noise[lower_tri_indices]
        
        # Use argpartition to find the indices of the top B informative pairs
        partitioned_indices = np.argpartition(-flat_I_noise, B-1)[:B]
        
        # Sort only the top B elements to get them in exact order
        top_indices_sorted = np.argsort(-flat_I_noise[partitioned_indices])

        # Get the sorted indices for the top pairs
        top_pairs_indices = (
            lower_tri_indices[0][partitioned_indices][top_indices_sorted],
            lower_tri_indices[1][partitioned_indices][top_indices_sorted]
        )
        
        # Form the array of pairs to return
        top_pairs = np.vstack(top_pairs_indices).T
        
        return top_pairs

    def compute_entropy(self, q):
        return -np.sum(q * np.log(q + 1e-10), axis=1)  # Add a small constant to prevent log(0)

    def sort_similarity_matrix(self, S):
        # Get the size of the matrix
        N = S.shape[0]

        # Create a list of tuples (value, (i, j))
        value_index_pairs = [(S[i, j], (i, j)) for i in range(N) for j in range(N)]

        # Sort the list in descending order of the values
        sorted_pairs = sorted(value_index_pairs, key=lambda x: x[0], reverse=True)

        # Print the sorted pairs and their values
        kk = 0
        for value, (i, j) in sorted_pairs:
            if i == j:
                continue
            #if value == 0:
                #continue
            print(f"Pair: ({i}, {j}), Value: {value}")
            kk += 1
            if kk > 10:
                break

        # Sort the list in descending order of the values
        sorted_pairs = sorted(value_index_pairs, key=lambda x: x[0], reverse=False)
        
        print("---------------------")

        # Print the sorted pairs and their values
        kk = 0
        for value, (i, j) in sorted_pairs:
            if i == j:
                continue
            print(f"Pair: ({i}, {j}), Value: {value}")
            kk += 1
            if kk > 10:
                break

    def compute_info_gain(self, h, q, S):
        if self.ac.clustering_alg != "mean_field":
            raise ValueError("Info gain is only defined for mean field clustering")

        N, K = q.shape
        I = np.zeros((N, N))

        print("HAASDSAD: ", H_C)
        #beta = self.ac.mean_field_beta
        beta = 1
        lmbda = 1

        q = softmax(beta*-h, axis=1)
        h = -np.dot(S, q)

        H_C = np.sum(self.compute_entropy(q))

        for x in range(N):
            for y in range(x):
                S_xy = S[x, y]

                # Compute h for P(C | e = 1)
                h_e1 = np.copy(h)
                h_e1[x, :] += S_xy * q[y, :] * lmbda - lmbda*q[y, :]
                h_e1[y, :] += S_xy * q[x, :] * lmbda - lmbda*q[x, :]
                #q_e1 = self.recompute_q(h_e1)
                q_e1 = softmax(beta*-h_e1, axis=1)
                H_C_e1 = np.sum(self.compute_entropy(q_e1))

                # Compute h for P(C | e = -1)
                h_e_minus_1 = np.copy(h)
                h_e_minus_1[x, :] += S_xy * q[y, :] * lmbda + lmbda*q[y, :]
                h_e_minus_1[y, :] += S_xy * q[x, :] * lmbda + lmbda*q[x, :]
                #q_e_minus_1 = self.recompute_q(h_e_minus_1)
                q_e_minus_1 = softmax(beta*-h_e_minus_1, axis=1)
                H_C_e_minus_1 = np.sum(self.compute_entropy(q_e_minus_1))

                # Compute P(e = 1)
                h_p_e1 = np.copy(h)
                h_p_e1[x, :] += S_xy * q[y, :] * lmbda - lmbda
                h_p_e1[y, :] += S_xy * q[x, :] * lmbda - lmbda
                #q_p_e1 = self.recompute_q(h_p_e1)
                q_p_e1 = softmax(beta*-h_p_e1, axis=1)
                P_e1 = np.sum(q_p_e1[x, :] * q_p_e1[y, :])
                #P_e1 = np.sum(q[x, :] * q[y, :])

                # Compute P(e = -1)
                #h_p_e_minus_1 = np.copy(h)
                #h_p_e_minus_1[x, :] += S_xy * q[y, :] + 1
                #h_p_e_minus_1[y, :] += S_xy * q[x, :] + 1
                #q_p_e_minus_1 = self.recompute_q(h_p_e_minus_1)
                #q_p_e_minus_1 = softmax(beta*-h_p_e_minus_1, axis=1)
                #P_e_minus_1 = np.sum(q_p_e_minus_1[x, :] * (1 - q_p_e_minus_1[y, :]) + q_p_e_minus_1[y, :] * (1 - q_p_e_minus_1[x, :]))
                P_e_minus_1 = 1 - P_e1 

                #print("P_e1: ", P_e1)
                #print("P_e_minus_1: ", P_e_minus_1)
                

                H_C_e = P_e1 * H_C_e1 + P_e_minus_1 * H_C_e_minus_1
                #print("H_C_e: ", H_C_e)
                I[x, y] = H_C - H_C_e
                I[y, x] = I[x, y]

        self.sort_similarity_matrix(I)
        #self.sort_similarity_matrix(self.ac.pairwise_similarities)

        return I
                
    def compute_maxmin(self):
        max_edges = self.ac.N
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        N = self.ac.N
        for i, j in zip(a, b):
            if self.violates_clustering(i, j):
                for k in range(0, N):
                    if k == i or k == j:
                        continue
                    sim_ij = self.ac.pairwise_similarities[i, j]
                    sim_ik = self.ac.pairwise_similarities[i, k]
                    sim_jk = self.ac.pairwise_similarities[j, k]
                    if self.triangle_is_bad(i, j, k):
                        sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                        smallest_sim = self.random_sort(np.array(sims))[0]
                        if smallest_sim == 0:
                            self.custom_informativeness[i, j] = np.abs(sim_ij)
                            self.custom_informativeness[j, i] = np.abs(sim_ij)
                        if smallest_sim == 1:
                            self.custom_informativeness[i, k] = np.abs(sim_ik)
                            self.custom_informativeness[k, i] = np.abs(sim_ik)
                        if smallest_sim == 2:
                            self.custom_informativeness[j, k] = np.abs(sim_jk)
                            self.custom_informativeness[k, j] = np.abs(sim_jk)
        return self.custom_informativeness

    def compute_maxexp(self):
        max_edges = self.ac.N
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        N = self.ac.N
        for i, j in zip(a, b):
            if self.violates_clustering(i, j):
                for k in range(0, N):
                    if k == i or k == j:
                        continue
                    sim_ij = self.ac.pairwise_similarities[i, j]
                    sim_ik = self.ac.pairwise_similarities[i, k]
                    sim_jk = self.ac.pairwise_similarities[j, k]

                    if self.triangle_is_bad(i, j, k):
                        expected_loss = self.min_triple_cost(i, j, k, beta=self.ac.beta) 
                        sims = [np.abs(sim_ij), np.abs(sim_ik), np.abs(sim_jk)]
                        smallest_sim = self.random_sort(np.array(sims))[0]
                        if smallest_sim == 0:
                            self.custom_informativeness[i, j] = np.maximum(expected_loss, self.custom_informativeness[i, j])
                            self.custom_informativeness[j, i] = np.maximum(expected_loss, self.custom_informativeness[j, i])
                        if smallest_sim == 1:
                            self.custom_informativeness[i, k] = np.maximum(expected_loss, self.custom_informativeness[i, k])
                            self.custom_informativeness[k, i] = np.maximum(expected_loss, self.custom_informativeness[k, i])
                        if smallest_sim == 2:
                            self.custom_informativeness[j, k] = np.maximum(expected_loss, self.custom_informativeness[j, k])
                            self.custom_informativeness[k, j] = np.maximum(expected_loss, self.custom_informativeness[k, j])
        return self.custom_informativeness

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
        b = np.random.random(arr.size)
        return np.lexsort((b, arr))

    