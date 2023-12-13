import numpy as np 
from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse


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
        elif acq_fn == "info_gain_object":
            self.info_matrix = self.compute_info_gain(self.ac.h, self.ac.q, self.ac.pairwise_similarities, mode="object")
        elif acq_fn == "info_gain_edge":
            self.info_matrix = self.compute_info_gain(self.ac.h, self.ac.q, self.ac.pairwise_similarities, mode="edge")
        elif acq_fn == "entropy":
            self.info_matrix = self.compute_entropy(self.ac.h, self.ac.q, self.ac.pairwise_similarities)
        elif acq_fn == "cluster_freq":
            self.info_matrix = self.compute_cluster_informativeness(-self.ac.feedback_freq)
            self.info_matrix = self.info_matrix - self.ac.feedback_freq
        elif acq_fn == "cluster_uncert":
            self.info_matrix = self.compute_cluster_informativeness(-np.abs(self.ac.pairwise_similarities))
            self.info_matrix = self.info_matrix - np.abs(self.ac.pairwise_similarities)
        elif acq_fn == "cluster_incon":
            self.info_matrix = self.compute_cluster_informativeness(self.ac.violations - self.ac.alpha*np.abs(self.ac.pairwise_similarities))
            self.info_matrix = self.info_matrix - np.abs(self.ac.pairwise_similarities)
        else:
            raise ValueError("Invalid acquisition function: {}".format(acq_fn))

        return self.select_edges(batch_size, self.info_matrix)
           
    def select_edges(self, batch_size, I):

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

    def sort_similarity_matrix(self, S):
        # Get the size of the matrix
        N = S.shape[0]

        # Create a list of tuples (value, (i, j))
        value_index_pairs = [(S[i, j], (i, j)) for i in range(N) for j in range(i)]

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

    def compute_entropy(self, h, q, S):
        if self.ac.clustering_alg != "mean_field":
            raise ValueError("Entropy only defined for mean field clustering")

        I = np.zeros((self.ac.N, self.ac.N))

        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S = sparse.csr_matrix(S)

        beta = self.ac.mean_field_beta
        #beta = self.ac.info_gain_beta
        #lmbda = self.ac.info_gain_lambda

        q = scipy_softmax(beta*-h, axis=1)
        h = -S.dot(q)

        P_e1_full = np.einsum('ik,jk->ij', q, q)
        P_e1 = P_e1_full[np.tril_indices(self.ac.N, k=-1)]
        P_e2 = 1 - P_e1
        entropies = scipy_entropy(np.vstack((P_e1, P_e2)), base=np.e, axis=0)
        I[np.tril_indices(self.ac.N, k=-1)] = entropies
        I += I.T

        return I

    def select_objects_info_gain(self, q, U_size, x, y, mode="uniform"):
        if mode == "uniform":
            return np.setdiff1d(np.random.choice(self.ac.N, U_size, replace=False), [x, y])
        elif mode == "entropy":
            # Exclude x and y from the computation
            indices = np.arange(self.ac.N) != x
            indices &= np.arange(self.ac.N) != y

            # Compute P(e_ix | Q) and P(e_iy | Q) for all i (except x and y)
            P_e_ix_Q = np.sum(q[indices, :] * q[x, :], axis=1)
            P_e_iy_Q = np.sum(q[indices, :] * q[y, :], axis=1)

            # Compute entropy of P(e_ix | Q) and P(e_iy | Q)
            entropy_e_ix_Q = scipy_entropy(np.stack((P_e_ix_Q, 1 - P_e_ix_Q), axis=1))
            entropy_e_iy_Q = scipy_entropy(np.stack((P_e_iy_Q, 1 - P_e_iy_Q), axis=1))

            # Compute the average entropy
            avg_entropy = (entropy_e_ix_Q + entropy_e_iy_Q) / 2

            # Rank objects based on average entropy and select top U_size objects
            ranked_indices = np.argsort(avg_entropy)[::-1][:U_size]

            # Extract the top U_size indices, excluding x and y
            top_U_indices = np.arange(self.ac.N)[indices][ranked_indices]
            return np.setdiff1d(top_U_indices, [x, y])
        elif mode == "uniform_varying":
            return np.array([])
        else:
            raise ValueError("Invalid mode (objects): {}".format(mode))

    def select_pairs_info_gain(self, mode):
        if mode == "uniform":
            lower_triangle_indices = np.tril_indices(self.ac.N, -1)
            inds = np.where(self.ac.feedback_freq[lower_triangle_indices] > 0)[0]
            num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else len(inds)
            inds = np.random.choice(inds, num_edges, replace=False)
            return np.stack((lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]), axis=-1)
        elif mode == "entropy":
            self.info_matrix = self.compute_entropy(self.ac.h, self.ac.q, self.ac.pairwise_similarities)
            lower_triangle_indices = np.tril_indices(self.ac.N, -1)
            inds = np.where(self.ac.feedback_freq[lower_triangle_indices] > 0)[0]
            num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else len(inds)
            return self.select_edges(num_edges, self.info_matrix)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def update_mean_fields(self, q_0, h_0, S, x, y, lmbda, L, U_size, G_size):
        h = np.copy(h_0)
        q = np.copy(q_0)
        q_prev = np.copy(q_0)

        U_size = int(U_size * self.ac.N)
        G_size = int(G_size * self.ac.N)

        # Initialize U^0 as an empty set
        U_prev = np.array([])

        U_all = np.array([])
        U_t = self.select_objects_info_gain(mode=self.ac.info_gain_object_mode, q=q, U_size=U_size, x=x, y=y)
        for t in range(1, L + 1):
            if t == 1:
                h[x, :] = h_0[x, :] + S[x, y] * q_0[y, :] - lmbda * q_0[y, :]
                h[y, :] = h_0[y, :] + S[y, x] * q_0[x, :] - lmbda * q_0[x, :]
            else:
                if self.ac.info_gain_object_mode == "uniform_varying":
                    U_t = np.setdiff1d(np.random.choice(self.ac.N, U_size, replace=False), [x, y])
                G = np.setdiff1d(np.random.choice(U_prev, np.minimum(G_size, len(U_prev)), replace=False), [x, y]).astype(int)

                U_all = np.union1d(U_all, U_t).astype(int)
                q[G] = scipy_softmax(-self.ac.mean_field_beta*h[G], axis=1)
                q[[x, y]] = scipy_softmax(-self.ac.mean_field_beta*h[[x, y]], axis=1)

                # update for objects in U_t
                G_xy = np.append(G, [x, y])
                delta_q_xy = (q_prev[G_xy, :] - q[G_xy, :])
                h[U_t, :] += S[U_t][:, G_xy].dot(delta_q_xy)

                # update for x and y
                delta_q = delta_q_xy[:-2, :] # remove last 2 elements (i.e. x and y)
                h[x, :] += S[x, G].dot(delta_q).reshape(h[x, :].shape)
                h[y, :] += S[y, G].dot(delta_q).reshape(h[y, :].shape)
                h[x, :] += lmbda * (q_prev[y, :] - q[y, :])
                h[y, :] += lmbda * (q_prev[x, :] - q[x, :])

                q_prev = np.copy(q)
                U_prev = U_t

        q_final = scipy_softmax(-h, axis=1)
        return q_final, U_all

    def compute_info_gain(self, h, q, S, mode="edge"):
        if self.ac.clustering_alg != "mean_field":
            raise ValueError("Info gain is only defined for mean field clustering")

        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S = sparse.csr_matrix(S)

        q = scipy_softmax(self.ac.mean_field_beta*-h, axis=1)
        h = -S.dot(q)

        #H_C = np.sum(scipy_entropy(q, axis=1))
        #H_e = np.sum(scipy_entropy(q, axis=1))
        W = self.select_pairs_info_gain(mode=self.ac.info_gain_pair_mode)
        I = np.zeros((self.ac.N, self.ac.N))

        # For each pair (x, y) in W
        for x, y in W:
            q_lambda, U_pos = self.update_mean_fields(
                q, h, S, x, y, self.ac.info_gain_lambda, L=self.ac.mf_iterations, U_size=self.ac.U_size, G_size=self.ac.G_size
            )
            q_minus_lambda, U_neg = self.update_mean_fields(
                q, h, S, x, y, -self.ac.info_gain_lambda, L=self.ac.mf_iterations, U_size=self.ac.U_size, G_size=self.ac.G_size
            )
            U = np.union1d(U_pos, U_neg)

            if mode == "object":
                P_e1 = np.sum(q[x, :] * q[y, :])
                P_e_minus_1 = 1 - P_e1 
                q_lambda_U = q_lambda[U, :]
                q_minus_lambda_U = q_minus_lambda[U, :]
                H_C_1 = np.sum(scipy_entropy(q_lambda_U, axis=1))
                H_C_2 = np.sum(scipy_entropy(q_minus_lambda_U, axis=1))
                H_C_e = P_e1 * H_C_1 + P_e_minus_1 * H_C_2
                I[x, y] = H_C_e
                I[y, x] = I[x, y]
            elif mode == "edge":
                I[x, y] = self.compute_info_gain_edge(q, q_lambda, q_minus_lambda, U, x, y)
                I[y, x] = I[x, y]
            else:
                raise ValueError("Invalid mode (compute_info_gain): {}".format(mode))
                
        return -I

    def compute_info_gain_edge(self, q, q_lambda, q_minus_lambda, U, x, y):
        U = np.array(list(U))

        p1 = np.sum(q[x] * q[y])
        p2 = 1 - p1

        lower_triangular = np.tril(np.einsum('ik,jk->ij', q_lambda[U, :], q_lambda[U, :]), k=-1)
        pair_probabilities = lower_triangular[lower_triangular != 0]
        pair_entropies_vectorized = scipy_entropy(np.vstack((pair_probabilities, 1 - pair_probabilities)), base=np.e, axis=0)
        H1 = np.sum(pair_entropies_vectorized)

        lower_triangular = np.tril(np.einsum('ik,jk->ij', q_minus_lambda[U, :], q_minus_lambda[U, :]), k=-1)
        pair_probabilities = lower_triangular[lower_triangular != 0]
        pair_entropies_vectorized = scipy_entropy(np.vstack((pair_probabilities, 1 - pair_probabilities)), base=np.e, axis=0)
        H2 = np.sum(pair_entropies_vectorized)


        I_U = p1 * H1 + p2 * H2

        return I_U

    def compute_cluster_informativeness(self, info_matrix):
        local_regions = []
        N = self.ac.N
        for i in range(len(self.ac.clustering)):
            for j in range(i + 1):
                cluster_i = self.ac.clustering[i]
                cluster_j = self.ac.clustering[j]
                if i != j:
                    pairwise_indices = list(product(cluster_i, cluster_j))
                    local_regions.append(pairwise_indices)
                else:
                    if len(cluster_i) == 1:
                        continue
                    pairwise_indices = list(combinations(cluster_i, 2))
                    local_regions.append(pairwise_indices)

        I = np.zeros((N, N))
        region_sums = []
        for region in local_regions:
            if len(region) == 0:
                continue

            row_indices, col_indices = zip(*region)
            region_elements = info_matrix[row_indices, col_indices]
            region_sum = np.sum(region_elements) / len(region)
            region_sums.append((region_sum, region))

        sorted_regions = sorted(region_sums, key=lambda x: x[0], reverse=True)
        decrement = 0
        for region_sum, region in sorted_regions:
            row_indices, col_indices = zip(*region)
            I[row_indices, col_indices] = region_sum - decrement
            I[col_indices, row_indices] = region_sum - decrement
            decrement += 10000
        return I

    def compute_maxmin(self):
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        if self.ac.num_maxmin_edges == 0:
            return self.custom_informativeness
        #max_edges = int(self.ac.num_maxmin_edges * self.ac.N)
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]

        if self.ac.num_maxmin_edges == -1:
            max_edges = self.ac.N
        else:
            num_violations = len(inds)
            max_edges = int(num_violations * self.ac.num_maxmin_edges)

        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
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
        #max_edges = int(self.ac.num_maxmin_edges * self.ac.N)
        self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
        if self.ac.num_maxmin_edges == 0:
            return self.custom_informativeness
        lower_triangle_indices = np.tril_indices(self.ac.N, -1)
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]

        if self.ac.num_maxmin_edges == -1:
            max_edges = self.ac.N
        else:
            num_violations = len(inds)
            max_edges = int(num_violations * self.ac.num_maxmin_edges)
        
        inds = np.where(np.abs(self.ac.violations[lower_triangle_indices]) > 0)[0]
        inds = np.random.choice(inds, np.min([max_edges, len(inds)]), replace=False)
        a, b = lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]
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
        probs = scipy_softmax(-beta*costs)
        expected_loss = np.sum(costs*probs)

        return expected_loss

    def triangle_is_bad(self, i, j, k):
        sim_ij = self.ac.pairwise_similarities[i, j]
        sim_ik = self.ac.pairwise_similarities[i, k]
        sim_jk = self.ac.pairwise_similarities[j, k]
        num_pos = 0

        if sim_ij >= 0:
            num_pos += 1

        if sim_ik >= 0:
            num_pos += 1

        if sim_jk >= 0:
            num_pos += 1

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

    