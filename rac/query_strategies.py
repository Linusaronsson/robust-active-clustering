import numpy as np 
from itertools import combinations, product

class QueryStrategy:
    def __init__(self, ac):
        self.ac = ac

    def select_batch(self, acq_fn, local_regions, batch_size):
        if type(local_regions) == str:
            if local_regions == "pairs":
                return np.array(self.select_pairs(acq_fn, "all_pairs", batch_size)), None
            elif local_regions == "triangles":
                pass
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
        pass

    def maxmin_ucb(self, local_region):
        pass

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