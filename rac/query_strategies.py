import numpy as np 
from itertools import combinations, product
from scipy.special import softmax

class QueryStrategy:
    def __init__(self, ac):
        self.ac = ac
        self.custom_informativeness = None

    def select_batch(self, acq_fn, local_regions, batch_size):

        if "maxmin" in acq_fn or "maxexp" in acq_fn:
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

    def compute_informativeness(self, acq_fn):
        if acq_fn == "maxmin":
            self.custom_informativeness = np.zeros((self.ac.N, self.ac.N), dtype=np.float32) 
            N = self.ac.N
            for i in range(0, N):
                for j in range(0, i):
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
            for i in range(0, N):
                for j in range(0, i):
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