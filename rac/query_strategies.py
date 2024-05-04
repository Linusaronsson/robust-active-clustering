import numpy as np 
#import cupy as np

from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse
from rac.correlation_clustering import mean_field_clustering, mean_field_clustering_torch
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
            self.info_matrix = self.compute_info_gain(S=self.ac.pairwise_similarities)
        elif acq_fn == "info_gain_pairs":
            self.info_matrix = self.compute_info_gain_pairs(S=self.ac.pairwise_similarities)
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
           
    def select_edges(self, batch_size, I, acq_noise=False, return_indices=False, use_tau=True):
        I_local = np.copy(I)
        tri_rows, tri_cols = np.tril_indices(n=I_local.shape[0], k=-1)
        informative_scores = I_local[tri_rows, tri_cols]

        if acq_noise and (self.ac.acq_fn in ["entropy", "info_gain_object", "info_gain_pairs", "info_gain_pairs_random"]):
            num_pairs = len(informative_scores)
            if self.ac.use_power:
                informative_scores[informative_scores < 0] = 1e-16
                informative_scores = np.log(informative_scores)
            power_beta = self.ac.power_beta
            informative_scores = informative_scores + scipy.stats.gumbel_r.rvs(loc=0, scale=1/power_beta, size=num_pairs, random_state=None)

        if use_tau:
            fq_flat = self.ac.feedback_freq[tri_rows, tri_cols]
            informative_scores[fq_flat > self.ac.tau] = -np.inf

        random_tie_breaker = np.random.rand(len(informative_scores))
        sorted_indices = np.lexsort((random_tie_breaker, -informative_scores))
        top_B_indices = sorted_indices[:batch_size]

        #random_tie_breaker = np.random.rand(len(informative_scores))
        #keys = np.vstack((random_tie_breaker, -informative_scores)).T  # Transpose to make keys columns
        #sorted_indices = np.lexsort(keys.T)  # Transpose back to make keys rows for sorting
        #top_B_indices = sorted_indices[:batch_size]

        if return_indices:
            return top_B_indices
        else:
            top_row_indices = tri_rows[top_B_indices]
            top_col_indices = tri_cols[top_B_indices]
            top_pairs = np.stack((top_row_indices, top_col_indices), axis=-1)
            return top_pairs
        
    def entropy_matrix(self, q, return_P=False):
        P = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P, 1 - P], axis=-1)
        I = scipy_entropy(distributions, base=np.e, axis=-1)

        if return_P:
            return I, P
        else:
            return I

    def _compute_mf(self, S, q=None, h=None):
        beta = self.ac.mean_field_beta
        
        if self.ac.sparse_sim_matrix and self.ac.repeat_id != 0:
            S = sparse.csr_matrix(S)

        if self.ac.repeat_id == 0:
            mf_alg = mean_field_clustering_torch
        else:
            mf_alg = mean_field_clustering

        if (self.ac.acq_fn in ["info_gain_object", "info_gain_pairs"]) and q is not None:
            n_iter = 50
        else:
            n_iter = self.ac.mf_iterations

        clust_sol, q, h = mf_alg(
            S=S,
            K=self.ac.num_clusters,
            beta=beta, 
            max_iter=n_iter, 
            tol=self.ac.conv_threshold, 
            noise=self.ac.mf_noise, 
            reinit=self.ac.reinit,
            predicted_labels=self.ac.clustering_solution,
            q=q,
            h=h
        )
        return q, h

    def compute_mean_field(self, S, q=None, h=None):
        q_avg_accumulator = np.zeros((self.ac.N, self.ac.num_clusters))
        for i in range(self.ac.num_mc_mf):
            q, h = self._compute_mf(S, q, h)
            q_avg_accumulator += q
        q_avg = q_avg_accumulator / self.ac.num_mc_mf
        q_avg = q_avg / np.sum(q_avg, axis=1, keepdims=True)
        return q_avg

    def compute_entropy(self, S, q=None):
        if q is None:
            #q  = self.compute_mean_field(S)
            q, h = self._compute_mf(S)
        I = self.entropy_matrix(q)
        return I

    def select_pairs_info_gain(self, mode, num_edges, q=None, acq_noise=False, return_indices=False, use_tau=True):
        if mode == "uniform":
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #info_matrix = np.random.rand(self.ac.N, self.ac.N) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            info_matrix = -self.ac.feedback_freq
        elif mode == "entropy":
            info_matrix = self.compute_entropy(S=self.ac.pairwise_similarities, q=q)
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return self.select_edges(num_edges, info_matrix, acq_noise=acq_noise, return_indices=return_indices, use_tau=use_tau)

    def compute_bald(self, S):
        q_avg = np.zeros((self.ac.N, self.ac.num_clusters))
        I_all = np.zeros((self.ac.N, self.ac.N))
        for _ in range(self.ac.num_mc_mf):
            q, h = self._compute_mf(S)
            q_avg += q / self.ac.num_mc_mf
            I_all += self.entropy_matrix(q) / self.ac.num_mc_mf
        q_avg = q_avg / np.sum(q_avg, axis=1, keepdims=True)
        I_total = self.entropy_matrix(q_avg)
        return I_total - I_all

    def compute_info_gain(self, S):
        q, h = self._compute_mf(S)
        num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else self.ac.n_edges
        num_edges = int(np.minimum(num_edges, self.ac.n_edges))
        W = self.select_pairs_info_gain(mode=self.ac.info_gain_pair_mode, q=q, num_edges=num_edges, acq_noise=self.ac.mf_acq_noise)
        I = np.zeros((self.ac.N, self.ac.N))
        H_0 = np.sum(scipy_entropy(q, axis=1))
        lmbda = self.ac.info_gain_lambda
        P = np.einsum('ik,jk->ij', q, q)
        for x, y in W:
            H_c_e = 0
            for outcome in [+1, -1]:
                S_new = np.copy(S)
                S_new[x, y] = outcome*lmbda 
                S_new[y, x] = outcome*lmbda
                np.fill_diagonal(S_new, 0)
                q_new, h_new = self._compute_mf(S_new, q, h)
                H_C = np.sum(scipy_entropy(q_new, axis=1))
                if outcome == 1:
                    prob = P[x, y]
                else:
                    prob = 1-P[x, y]
                H_c_e += prob * H_C
            I[x, y] = H_0-H_c_e
            I[y, x] = I[x, y]
        return I

    def compute_info_gain_pairs(self, S):
        q, h = self._compute_mf(S)
        num_edges = int(self.ac.num_edges_info_gain*self.ac.N) if self.ac.num_edges_info_gain > 0 else self.ac.n_edges
        num_edges = int(np.minimum(num_edges, self.ac.n_edges))
        W = self.select_pairs_info_gain(mode=self.ac.info_gain_pair_mode, q=q, num_edges=num_edges, acq_noise=self.ac.mf_acq_noise)
        I = np.zeros((self.ac.N, self.ac.N))
        lmbda = self.ac.info_gain_lambda
        I_pairs, P = self.entropy_matrix(q, return_P=True)
        lower_triangular = np.tril(I_pairs, k=-1)
        H_0 = np.sum(lower_triangular)
        for x, y in W:
            H_c_e = 0
            for outcome in [+1, -1]:
                S_new = np.copy(S)
                S_new[x, y] = outcome*lmbda 
                S_new[y, x] = outcome*lmbda
                np.fill_diagonal(S_new, 0)
                q_new, h_new = self._compute_mf(S_new, q, h)
                I_pairs, _ = self.entropy_matrix(q_new, return_P=True)
                lower_triangular = np.tril(I_pairs, k=-1)
                H_C = np.sum(lower_triangular)
                if outcome == 1:
                    prob = P[x, y]
                else:
                    prob = 1-P[x, y]
                H_c_e += prob * H_C
            I[x, y] = H_0-H_c_e
            I[y, x] = I[x, y]
        return I

    # random pairs
    def compute_info_gain_pairs_random(self, S):
        q, h = self._compute_mf(S)
        I, P = self.entropy_matrix(q, return_P=True)
        I_all = np.zeros((self.ac.N, self.ac.N))

        if self.ac.r < 1:
            num_edges = int(self.ac.r * self.ac.n_edges)
        else:
            num_edges = self.ac.r

        num_edges = int(np.minimum(num_edges, self.ac.n_edges))

        i_lower, j_lower = np.tril_indices(self.ac.N, -1)
        for i in range(self.ac.num_mc_samples):
            selected = self.select_pairs_info_gain(
                mode=self.ac.info_gain_pair_mode, q=q, num_edges=num_edges, return_indices=True, use_tau=False, acq_noise=True
            )
            for j in range(self.ac.num_mc):
                S_new = np.copy(S)
                selected_i, selected_j = i_lower[selected], j_lower[selected]
                random_values = np.random.rand(num_edges)
                selected_values = np.where(random_values < P[selected_i, selected_j], 1, -1)
                S_new[selected_i, selected_j] = selected_values
                S_new[selected_j, selected_i] = selected_values
                np.fill_diagonal(S_new, 0)
                q_new, h_new = self._compute_mf(S_new, q, h)
                I_all += self.entropy_matrix(q_new) / self.ac.num_mc
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

    