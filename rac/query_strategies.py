import numpy as np 
from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse
from rac.correlation_clustering import mean_field_clustering
import scipy

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
            self.info_matrix = self.compute_info_gain(S=self.ac.pairwise_similarities, mode="object")
        elif acq_fn == "info_gain_object":
            self.info_matrix = self.compute_info_gain(S=self.ac.pairwise_similarities, mode="object")
        elif acq_fn == "info_gain_objects_all":
            self.info_matrix = self.compute_info_gain_objects_all(S=self.ac.pairwise_similarities)
        elif acq_fn == "info_gain_objects_random":
            self.info_matrix = self.compute_info_gain_objects_random(S=self.ac.pairwise_similarities)
        elif acq_fn == "info_gain_pairs_all":
            self.info_matrix = self.compute_info_gain_pairs_all(S=self.ac.pairwise_similarities)
        elif acq_fn == "info_gain_pairs_random":
            self.info_matrix = self.compute_info_gain_pairs_random(S=self.ac.pairwise_similarities)
        elif acq_fn == "entropy":
            self.info_matrix = self.compute_entropy(S=self.ac.pairwise_similarities)
        elif acq_fn == "bald":
            self.info_matrix = self.compute_bald(S=self.ac.pairwise_similarities)
        elif acq_fn == "cluster_freq":
            self.info_matrix = self.compute_cluster_informativeness(-self.ac.feedback_freq)
            self.info_matrix = self.info_matrix - self.ac.feedback_freq
        elif acq_fn == "cluster_uncert":
            self.info_matrix = self.compute_cluster_informativeness(-np.abs(self.ac.pairwise_similarities))
            self.info_matrix = self.info_matrix - np.abs(self.ac.pairwise_similarities)
        elif acq_fn == "cluster_incon":
            #if np.random.rand() < self.ac.eps:
                #self.info_matrix = -self.ac.feedback_freq
            #else:
            self.info_matrix1 = self.compute_cluster_informativeness(self.ac.violations)# - self.ac.alpha*np.abs(self.ac.pairwise_similarities))
            self.info_matrix2 = self.compute_cluster_informativeness(np.abs(self.ac.pairwise_similarities))
            self.info_matrix = self.ac.alpha1 * self.info_matrix1 - self.ac.alpha2*self.info_matrix2
            self.info_matrix = self.info_matrix - self.ac.alpha3*np.abs(self.ac.pairwise_similarities)

            #self.info_matrix = self.compute_cluster_informativeness(self.ac.violations - self.ac.alpha*np.abs(self.ac.pairwise_similarities))
            ##self.info_matrix2 = self.compute_cluster_informativeness(np.abs(self.ac.pairwise_similarities))
            ##self.info_matrix = self.info_matrix1 - self.ac.alpha*self.info_matrix2
            #self.info_matrix = self.info_matrix - np.abs(self.ac.pairwise_similarities)
        else:
            raise ValueError("Invalid acquisition function: {}".format(acq_fn))

        return self.select_edges(batch_size, self.info_matrix, acq_noise=self.ac.acq_noise)
           
    def select_edges(self, batch_size, I, acq_noise=False):
        I_local = np.copy(I)
        tri_rows, tri_cols = np.tril_indices(n=I_local.shape[0], k=-1)
        informative_scores = I_local[tri_rows, tri_cols]
        if acq_noise and (self.ac.acq_fn == "entropy" or self.ac.acq_fn == "info_gain_object"):
            num_pairs = len(informative_scores)
            if self.ac.fix_neg:
                informative_scores += np.abs(np.min(informative_scores)) + 1e-10
            else:
                informative_scores[informative_scores < 0] = 1e-10
            #print("max: ", np.max(informative_scores))
            #print("min: ", np.min(informative_scores))
            #informative_scores = np.abs(informative_scores)
            if self.ac.use_power:
                informative_scores = np.log(informative_scores)
            #print("max log: ", np.max(informative_scores))
            #print("min log: ", np.min(informative_scores))
            power_beta = 1
            informative_scores = informative_scores + scipy.stats.gumbel_r.rvs(loc=0, scale=1/power_beta, size=num_pairs, random_state=None)

        #top_B_indices = np.argpartition(informative_scores, -batch_size)[-batch_size:]

        # ensure that the query is not repeated
        fq_flat = self.ac.feedback_freq[tri_rows, tri_cols]
        informative_scores[fq_flat > self.ac.tau] = -np.inf

        random_tie_breaker = np.random.rand(len(informative_scores))
        sorted_indices = np.lexsort((random_tie_breaker, -informative_scores))
        top_B_indices = sorted_indices[:batch_size]

        top_row_indices = tri_rows[top_B_indices]
        top_col_indices = tri_cols[top_B_indices]
        top_pairs = np.stack((top_row_indices, top_col_indices), axis=-1)
        return top_pairs
        
    def compute_entropy(self, S, h=None, q=None):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S = sparse.csr_matrix(S)

        q_avg_accumulator = np.zeros((self.ac.N, self.ac.num_clusters))
        for i in range(self.ac.num_mc_entropy):
            clust_sol, q, h = mean_field_clustering(
                S=S, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                predicted_labels=self.ac.clustering_solution
            )
            q_avg_accumulator += q
        q_avg = q_avg_accumulator / self.ac.num_mc_entropy

        P_e1_full = np.einsum('ik,jk->ij', q_avg, q_avg)
        distributions = np.stack([P_e1_full, 1 - P_e1_full], axis=-1)
        I = scipy_entropy(distributions, base=np.e, axis=-1)
        return I

    def select_pairs_info_gain(self, mode, q, h):
        if mode == "uniform":
            lower_triangle_indices = np.tril_indices(self.ac.N, -1)
            inds = np.where(self.ac.feedback_freq[lower_triangle_indices] > 0)[0]
            num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else len(inds)
            inds = np.random.choice(inds, num_edges, replace=False)
            return np.stack((lower_triangle_indices[0][inds], lower_triangle_indices[1][inds]), axis=-1)
        elif mode == "entropy":
            info_matrix = self.compute_entropy(S=self.ac.pairwise_similarities, h=h, q=q)
            lower_triangle_indices = np.tril_indices(self.ac.N, -1)
            inds = np.where(self.ac.feedback_freq[lower_triangle_indices] > 0)[0]
            num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else len(inds)
            return self.select_edges(num_edges, info_matrix, use_grumbel=False)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def compute_bald(self, S):
        q_avg_accumulator = np.zeros((self.ac.N, self.ac.num_clusters))
        I_all = np.zeros((self.ac.N, self.ac.N))
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S = sparse.csr_matrix(S)
        # Iterate over the m runs of the clustering algorithm
        for _ in range(self.ac.num_mc_entropy):
            _, q, h = mean_field_clustering(
                S=S, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                predicted_labels=self.ac.clustering_solution
            )
            
            # Update the running total for q_avg
            q_avg_accumulator += q

            P_e1_full = np.einsum('ik,jk->ij', q, q)
            distributions = np.stack([P_e1_full, 1 - P_e1_full], axis=-1)

            # Calculating entropy for each distribution
            I_all += scipy_entropy(distributions, base=np.e, axis=-1)
        
        # Average the accumulated q values to get q_avg
        q_avg = q_avg_accumulator / self.ac.num_mc_samples
        q_avg = q_avg / np.sum(q_avg, axis=1, keepdims=True)
        P_e1_full = np.einsum('ik,jk->ij', q_avg, q_avg)
        distributions = np.stack([P_e1_full, 1 - P_e1_full], axis=-1)

        I_total = scipy_entropy(distributions, base=np.e, axis=-1)

        # Calculating entropy for each distribution
        I_all = I_all / self.ac.num_mc_samples
        
        # Return the difference between the entropy of q_avg and the average of entropies of individual q's
        return I_total - I_all
            

    def compute_info_gain(self, S, mode="object", h=None, q=None):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S_ = sparse.csr_matrix(S)

        if h is None:
            clust_sol, q, h = mean_field_clustering(
                S=S_, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                predicted_labels=self.ac.clustering_solution
            )
            
        W = self.select_pairs_info_gain(mode=self.ac.info_gain_pair_mode, q=q, h=h)
        I = np.zeros((self.ac.N, self.ac.N))
        H_0 = np.sum(scipy_entropy(q, axis=1))
        lmbda = self.ac.info_gain_lambda

        H_c_e = 0
        P = np.einsum('ik,jk->ij', q, q)
        # For each pair (x, y) in W
        for x, y in W:
            for outcome in [+1, -1]:
                S_new = np.copy(S)
                S_new[x, y] = S_new[y, x] = outcome*lmbda
                
                np.fill_diagonal(S_new, 0)
                if self.ac.sparse_sim_matrix and not sparse.issparse(S_new):
                    S_new = sparse.csr_matrix(S_new)
                _, q_new, h_new = mean_field_clustering(
                    S=S_new, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                    predicted_labels=self.ac.clustering_solution, h=h, q=q
                )
                H_C = np.sum(scipy_entropy(q_new, axis=1))
                prob = P[x, y] if outcome == 1 else 1-P[x, y]
                H_c_e += prob * H_C

            I[x, y] = H_0-H_c_e
            I[y, x] = I[x, y]
        return I

    # random pairs
    def compute_info_gain_pairs_random(self, S):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S_ = sparse.csr_matrix(S)

        clust_sol, q, h = mean_field_clustering(
            S=S_, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.ac.clustering_solution
        )

        P = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P, 1 - P], axis=-1)

        # Calculating entropy for each distribution
        I = scipy_entropy(distributions, base=np.e, axis=-1)
            
        I_all = np.zeros((self.ac.N, self.ac.N))
        #print("starting...")

        if self.ac.r < 1:
            num_elements_to_select = int(self.ac.r * self.ac.n_edges)
        else:
            num_elements_to_select = self.ac.r
        num_elements_to_select = int(np.minimum(num_elements_to_select, self.ac.n_edges))


        i_lower, j_lower = np.tril_indices(self.ac.N, -1)
        for i in range(self.ac.num_mc_samples):
            selected = np.random.choice(i_lower.shape[0], size=num_elements_to_select, replace=False)
            for j in range(self.ac.num_mc):
                S_new = np.copy(S)

                # Get the selected indices
                selected_i, selected_j = i_lower[selected], j_lower[selected]

                # Sample binary values for the selected indices
                random_values = np.random.rand(num_elements_to_select)
                selected_values = np.where(random_values < P[selected_i, selected_j], 1, -1)

                # Update the S matrix
                S_new[selected_i, selected_j] = selected_values
                S_new[selected_j, selected_i] = selected_values
                np.fill_diagonal(S_new, 0)
                if self.ac.sparse_sim_matrix and not sparse.issparse(S_new):
                    S_new = sparse.csr_matrix(S_new)
                _, q_new, h_new = mean_field_clustering(
                    S=S_new, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                    predicted_labels=self.ac.clustering_solution, h=h, q=q
                )
                P_e1_full = np.einsum('ik,jk->ij', q_new, q_new)
                distributions = np.stack([P_e1_full, 1 - P_e1_full], axis=-1)

                # Calculating entropy for each distribution
                I_all += scipy_entropy(distributions, base=np.e, axis=-1)/self.ac.num_mc
        
        # For each pair (x, y) in W
        
        #print("done...")
        return I - (I_all/self.ac.num_mc_samples)

    # all pairs
    def compute_info_gain_pairs_all(self, S):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S_ = sparse.csr_matrix(S)

        clust_sol, q, h = mean_field_clustering(
            S=S_, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.ac.clustering_solution
        )

        P = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P, 1 - P], axis=-1)

        # Calculating entropy for each distribution
        I = scipy_entropy(distributions, base=np.e, axis=-1)
        I_all = np.zeros((self.ac.N, self.ac.N))
    
        # Efficiently generate indices for the lower triangular part of the matrix, excluding the diagonal
        tri_indices = np.tril_indices(self.ac.N, k=-1)
        
        # Randomly sample pairs from these indices
        #if self.ac.r < 1:
        #    num_elements_to_select = int(self.ac.r * self.ac.n_edges)
        #else:
        #    num_elements_to_select = self.ac.r

        num_elements_to_select = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else self.ac.n_edges

        num_elements_to_select = int(np.minimum(num_elements_to_select, self.ac.n_edges))
        sampled_pairs = np.random.choice(len(tri_indices[0]), size=num_elements_to_select, replace=False)
        
        # Process each sampled pair
        for idx in sampled_pairs:
            i, j = tri_indices[0][idx], tri_indices[1][idx]
            
            # Iterate over both outcomes for the pair
            for outcome in [+1, -1]:
                # Initialize S_temp to ensure it's symmetric and has zeros on the diagonal after updates
                S_new = np.copy(S)
                S_new[i, j] = S_new[j, i] = outcome
                
                # Assuming a function that recalculates 'q' given the updated similarity matrix S_temp
                # This function involves re-running the clustering algorithm for the updated S_temp
                np.fill_diagonal(S_new, 0)
                if self.ac.sparse_sim_matrix and not sparse.issparse(S_new):
                    S_new = sparse.csr_matrix(S_new)
                _, q_new, h_new = mean_field_clustering(
                    S=S_new, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                    predicted_labels=self.ac.clustering_solution, h=h, q=q
                )
                
                # Calculate P_new and its entropy
                P_new = np.einsum('ik,jk->ij', q_new, q_new)
                distributions_new = np.stack([P_new, 1 - P_new], axis=-1)
                I_new = scipy_entropy(distributions_new, base=np.e, axis=-1)
                
                # Weight the added entropy by the probability of the outcome
                probability_weight = P[i, j] if outcome == +1 else (1 - P[i, j])
                I_all += probability_weight * I_new
        
        
        # Return the difference between the initial entropy matrix I and the accumulated one I_all
        return I - (I_all/num_elements_to_select)
        
    # all objects
    def compute_info_gain_objects_all(self, S):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S_ = sparse.csr_matrix(S)

        clust_sol, q, h = mean_field_clustering(
            S=S_, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.ac.clustering_solution
        )

        P = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P, 1 - P], axis=-1)

        # Calculating entropy for each distribution
        I = scipy_entropy(distributions, base=np.e, axis=-1)
        I_all = np.zeros((self.ac.N, self.ac.N))

        for i in range(self.ac.N):  # Iterate over each data point
            for k in range(q.shape[1]):  # Iterate over each cluster
                # Create a temporary similarity matrix copy to modify
                S_new = S.copy()

                # Update the similarity matrix for the current data point and cluster
                in_cluster = np.where(self.ac.clustering_solution == k)[0]
                not_in_cluster = np.where(self.ac.clustering_solution != k)[0]

                S_new[i, in_cluster] = +1
                S_new[i, not_in_cluster] = -1

                S_new[in_cluster, i] = +1
                S_new[not_in_cluster, i] = -1

                # Assume a function that recalculates 'q' given the updated similarity matrix
                # This function would typically involve re-running the clustering algorithm
                # For the sake of example, let's call this function recalculate_q()
                np.fill_diagonal(S_new, 0)
                if self.ac.sparse_sim_matrix and not sparse.issparse(S_new):
                    S_new = sparse.csr_matrix(S_new)
                _, q_new, h_new = mean_field_clustering(
                    S=S_new, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                    predicted_labels=self.ac.clustering_solution, h=h, q=q
                )

                # Calculate P_new and its entropy
                P_new = np.einsum('ik,jk->ij', q_new, q_new)
                distributions_new = np.stack([P_new, 1 - P_new], axis=-1)
                I_new = scipy_entropy(distributions_new, base=np.e, axis=-1)

                # Update I_all weighted by the probability q[i, k]
                I_all += q[i, k] * I_new

        # Return the difference between the initial entropy matrix and the accumulated one
        return I - (I_all/self.ac.N)

    # random objects
    def compute_info_gain_objects_random(self, S):
        if self.ac.sparse_sim_matrix and not sparse.issparse(S):
            S_ = sparse.csr_matrix(S)

        clust_sol, q, h = mean_field_clustering(
            S=S_, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.ac.clustering_solution
        )

        P = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P, 1 - P], axis=-1)

        # Calculating entropy for each distribution
        I = scipy_entropy(distributions, base=np.e, axis=-1)
        I_all = np.zeros((self.ac.N, self.ac.N))

        if self.ac.r < 1:
            num_elements_to_select = int(self.ac.r * self.ac.N)
        else:
            num_elements_to_select = self.ac.r
        num_elements_to_select = np.minimum(num_elements_to_select, self.ac.N)

        for iteration in range(self.ac.num_mc_samples):
            # Sample a random subset of data points
            sampled_indices = np.random.choice(self.ac.N, size=num_elements_to_select, replace=False)

            for mc_step in range(self.ac.num_mc):
                # Vectorized sampling of cluster outcomes for each data point in the subset
                # Generate cumulative probabilities for vectorized sampling
                cumulative_probabilities = np.cumsum(q[sampled_indices], axis=1)
                random_values = np.random.rand(num_elements_to_select, 1)
                outcomes = (random_values < cumulative_probabilities).argmax(axis=1)

                S_new = np.copy(S)  # Reset S_temp for each Monte Carlo step
                for i, sampled_cluster in zip(sampled_indices, outcomes):
                    # Determine data points in the same cluster as the sampled outcome for i
                    in_cluster = np.where(self.ac.clustering_solution == sampled_cluster)[0]
                    not_in_cluster = np.where(self.ac.clustering_solution != sampled_cluster)[0]

                    # Update similarities based on the current clustering
                    S_new[i, in_cluster] = 1
                    S_new[in_cluster, i] = 1  # Ensure symmetry
                    S_new[i, not_in_cluster] = -1
                    S_new[not_in_cluster, i] = -1  # Ensure symmetry

                np.fill_diagonal(S_new, 0)
                if self.ac.sparse_sim_matrix and not sparse.issparse(S_new):
                    S_new = sparse.csr_matrix(S_new)
                _, q_new, h_new = mean_field_clustering(
                    S=S_new, K=self.ac.num_clusters, betas=[self.ac.mean_field_beta], max_iter=100, tol=1e-10, 
                    predicted_labels=self.ac.clustering_solution, h=h, q=q
                )

                # Calculate P_new and entropy in a vectorized manner
                P_new = np.einsum('ik,jk->ij', q_new, q_new)
                distributions_new = np.stack([P_new, 1 - P_new], axis=-1)
                I_new = scipy_entropy(distributions_new, base=np.e, axis=-1)

                I_all += I_new / self.ac.num_mc

        # Calculate the initial entropy matrix for comparison
        distributions = np.stack([P, 1 - P], axis=-1)
        I = scipy_entropy(distributions, base=np.e, axis=-1)

        return I - (I_all/self.ac.num_mc_samples)

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
            I[row_indices, col_indices] = region_sum
            I[col_indices, row_indices] = region_sum


        #sorted_regions = sorted(region_sums, key=lambda x: x[0], reverse=True)
        #decrement = 0
        #for region_sum, region in sorted_regions:
        #    row_indices, col_indices = zip(*region)
        #    I[row_indices, col_indices] = region_sum - decrement
        #    I[col_indices, row_indices] = region_sum - decrement
        #    decrement += 10000
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

        if sim_ij > 0:
            num_pos += 1

        if sim_ik > 0:
            num_pos += 1

        if sim_jk > 0:
            num_pos += 1

        return num_pos == 2

    def in_same_cluster(self, o1, o2):
        return self.ac.clustering_solution[o1] == self.ac.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] > 0) or \
                (self.in_same_cluster(o1, o2) and self.ac.pairwise_similarities[o1, o2] < 0)
    
    def random_sort(self, arr, ascending=True):
        if not ascending:
            arr = -arr
        b = np.random.random(arr.size)
        return np.lexsort((b, arr))

    