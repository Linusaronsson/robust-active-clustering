import time
import math

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, accuracy_score
import itertools

import numpy as np
#import cupy as np

from scipy.spatial import distance

from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier

from rac.correlation_clustering import max_correlation, max_correlation_dynamic_K, mean_field_clustering, mean_field_clustering_torch
from rac.query_strategies import QueryStrategy
from rac.experiment_data import ExperimentData

from scipy import sparse
from scipy.stats import entropy as scipy_entropy

#import warnings
#warnings.filterwarnings("once") 

class ActiveClustering:
    def __init__(self, X, Y, repeat_id, initial_clustering_solution=None, **kwargs):
        self.__dict__.update(kwargs)

        self.X, self.Y = X, Y
        self.repeat_id = repeat_id
        self.initial_clustering_solution = initial_clustering_solution
        self.ac_data = ExperimentData(Y, repeat_id, **kwargs)
        self.qs = QueryStrategy(self)

        self.clustering = None
        self.clustering_solution = None
        self.N = len(self.Y)
        self.n_edges = (self.N*(self.N-1))/2

        #if self.N <= 500:
            #self.batch_size == 0.0025

        if self.batch_size < 1:
            self.batch_size = math.ceil(self.n_edges * self.batch_size)

        np.random.seed(self.repeat_id+self.seed+317421)

        if self.tau == -1:
            self.tau = np.inf
        else:
            self.tau = self.tau

        if self.acq_fn not in ["info_gain_object", "info_gain_pairs_random", "info_gain_pairs"]:
            self.power_beta = 1
            #self.tau = 3


        if self.acq_fn in ["info_gain_object", "info_gain_pairs"]:
            self.info_gain_pair_mode = "entropy"

    def run_AL_procedure(self):
        self.start_time = time.time()
        self.initialize_ac_procedure()
        self.store_experiment_data(initial=True)

        if self.repeat_id == 0:
            while True:
                clust_sol, q, h = mean_field_clustering_torch(
                    S=self.pairwise_similarities, K=self.num_clusters,
                    beta=self.mean_field_beta, 
                    max_iter=self.mf_iterations, 
                    tol=self.conv_threshold, 
                    noise=0, 
                    reinit=True,
                    predicted_labels=self.clustering_solution,
                    q=None,
                    h=None
                )


        perfect_rand_count = 0
        early_stopping_count = 0
        old_rand = adjusted_rand_score(self.Y, self.clustering_solution)
        total_queries = 0

        if self.acq_fn in ["QECC"]:
            stopping_criteria = self.batch_size * 250
        if self.acq_fn in ["nCOBRAS", "COBRAS"]:
            stopping_criteria = self.batch_size * 200
        else:
            stopping_criteria = self.n_edges

        #early_stopping = (self.n_edges/self.batch_size)*1
        early_stopping = (self.n_edges/self.batch_size)*0.55
        #ii = 0

        ii = 1
        while total_queries < stopping_criteria: 
        #while True:
            self.start = time.time()
            ii += 1
            if self.acq_fn == "QECC":
                self.clustering = self.QECC_heur(self.batch_size * ii)
                self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
                self.clustering_solution = self.clustering_solution_from_clustering(self.clustering)
            elif self.acq_fn == "nCOBRAS":
                if (ii % 1) == 0: 
                    try:
                        self.clustering_solution = self.COBRAS(self.batch_size * ii, mode="nCOBRAS") 
                        self.clustering_solution = np.array(self.clustering_solution, dtype=np.uint32)
                        self.clustering, num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
                        self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
                    except Exception:
                        print("nCOBRAS failed")
                        break
            elif self.acq_fn == "COBRAS":
                if (ii % 10) == 0: 
                    try:
                        self.clustering_solution = self.COBRAS(self.batch_size * ii, mode="COBRAS") 
                        self.clustering_solution = np.array(self.clustering_solution, dtype=np.uint32)
                        self.clustering, num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
                        self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
                    except Exception:
                        print("COBRAS failed")
                        break
            else:
                start_selct_batch = time.time()
                self.edges = self.qs.select_batch(
                    acq_fn=self.acq_fn,
                    batch_size=self.batch_size
                )
                self.time_select_batch = time.time() - start_selct_batch
                #print("TIME SELECT BATCH: ", self.time_select_batch)

                #if len(objects) == 1:
                    #raise ValueError("Singleton cluster in run_AL_procedure(...)")

                if len(self.edges) != self.batch_size:
                    raise ValueError("Num queried {} not equal to query size {}!!".format(len(self.edges[0]), self.batch_size))
                
                self.num_repeat_queries = 0
                for ind1, ind2 in self.edges:
                    self.update_similarity(ind1, ind2)

                time_clustering = time.time()
                self.update_clustering() 
                self.time_clustering = time.time() - time_clustering
                #print("TIME CLUSTERING: ", self.time_clustering)
                #print("time_nn: ", time.time() - time_clustering)


                for i in range(self.N):
                    for j in range(i):
                        if self.violates_clustering(i, j):
                            self.violations[i, j] = np.abs(self.pairwise_similarities[i, j])
                            self.violations[j, i] = np.abs(self.pairwise_similarities[j, i])
                        else:
                            self.violations[i, j] = 0
                            self.violations[j, i] = 0

            if ii >= self.start_inferring:
                self.infer_similarities()

            self.store_experiment_data()
            total_queries += self.batch_size
            self.total_time_elapsed = time.time() - self.start_time

            num_hours = self.total_time_elapsed / 3600
            if self.acq_fn in ["info_gain_object", "info_gain_pairs"]:
                max_hours = 47
            else:
                max_hours = 5

            if self.dataset_name == "synthetic":
                max_hours = 5

            if num_hours > max_hours:
                break

            if self._verbose:
                print("iteration: ", ii)
                print("prop_queried: ", total_queries/self.n_edges)
                print("feedback_freq max: ", np.max(self.feedback_freq))
                print("rand score: ", adjusted_rand_score(self.Y, self.clustering_solution))
                print("time: ", time.time()-self.start)
                if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                    print("num queries: ", len(self.edges))
                    print("TIME SELECT BATCH: ", self.time_select_batch)
                    print("TIME CLUSTERING: ", self.time_clustering)
                print("num clusters: ", self.num_clusters)
                #print("-----------------")
                
            #ii += 1
            current_rand = adjusted_rand_score(self.Y, self.clustering_solution)

            #if current_rand > 0.98:
                #break
            
            if current_rand == old_rand:
                early_stopping_count += 1
            else:
                early_stopping_count = 0
            
            if current_rand >= 1:
                perfect_rand_count += 1
            else:
                perfect_rand_count = 0
            
            old_rand = current_rand

            if early_stopping_count > early_stopping or perfect_rand_count > 2:
                break
        
        return self.ac_data

    def QECC_heur(self, num_queries):
        nodes = set(np.arange(0, self.N).tolist())
        vertices = nodes.copy()
        query_budget = num_queries
        clusters = []
        while len(vertices) > 1 and query_budget > (len(vertices) - 1):
            unique_pairs = list(itertools.combinations(vertices, 2))
            np.random.shuffle(unique_pairs)
            for (ind1, ind2) in unique_pairs:
                query = self.get_similarity(ind1, ind2)
                query_budget -= 1
                if query >= 0:
                    break

            cluster = {ind1, ind2}
            for nd in vertices - {ind1, ind2}:
                if self.get_similarity(ind2, nd) > 0:
                    cluster.add(nd)
            clusters.append(list(cluster))
            query_budget = query_budget - len(vertices) + 2
            vertices = vertices - cluster
        for nd in vertices:
            clusters.append([nd])
        return clusters

    def QECC(self, num_queries):
        nodes = set(np.arange(0, self.N).tolist())
        vertices = nodes.copy()
        query_budget = num_queries
        clusters = []
        while len(vertices) > 1 and query_budget > (len(vertices) - 1):
            ind1 = np.random.choice(len(vertices), 1, replace=False)[0]
            cluster = {ind1}
            for nd in vertices - {ind1}:
                if self.get_similarity(ind1, nd) > 0:
                    cluster.add(nd)
            clusters.append(list(cluster))
            query_budget = query_budget - len(vertices) + 1
            vertices = vertices - cluster
        for nd in vertices:
            clusters.append([nd])
        return clusters

    def COBRAS(self, num_queries, mode="COBRAS"):
        if mode == "COBRAS":
            noisy_querier = ProbabilisticNoisyQuerier(
                None, self.Y, (self.noise_level/2), num_queries,
                random_seed=self.repeat_id+self.seed+3892
            )
            clusterer = COBRAS(correct_noise=False, seed=self.repeat_id+self.seed+4873)
        elif mode == "nCOBRAS":
            noise_level_cobras = (self.noise_level/2)
            if self.noise_level == 0:
                noise_level_cobras = 0.05
            noisy_querier = ProbabilisticNoisyQuerier(
                None, self.Y, (self.noise_level/2), num_queries,
                random_seed=self.repeat_id+self.seed+3892
            )
            clusterer = COBRAS(
                correct_noise=True, noise_probability=noise_level_cobras,
                certainty_threshold=0.95, minimum_approximation_order=2,
                maximum_approximation_order=3,
                seed=self.repeat_id+self.seed+4873
            )
        all_clusters, runtimes, *_ = clusterer.fit(self.X, -1, None, noisy_querier)
        return all_clusters[-1]

    def store_experiment_data(self, initial=False):
        self.ac_data.rand.append(adjusted_rand_score(self.Y, self.clustering_solution))
        self.ac_data.ami.append(adjusted_mutual_info_score(self.Y, self.clustering_solution))
        self.ac_data.v_measure.append(v_measure_score(self.Y, self.clustering_solution))
        self.ac_data.num_clusters.append(self.num_clusters)
                
        if initial:
            self.ac_data.Y = self.Y
            #self.ac_data.num_queries.append(0)
            self.ac_data.time.append(0.0)
            self.ac_data.time_select_batch.append(0.0)
            self.ac_data.time_update_clustering.append(0.0)
            self.ac_data.num_repeat_queries.append(0)
            self.ac_data.num_violations.append(0)
        else:
            count_non_zero_lower = np.count_nonzero(np.tril(self.violations, k=-1))
            self.ac_data.num_violations.append(count_non_zero_lower)
            self.ac_data.num_repeat_queries.append(self.num_repeat_queries)
            #self.ac_data.num_queries.append(self.batch_size) 
            time_now = time.time() 
            self.ac_data.time.append(time_now - self.start)
            if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                self.ac_data.time_select_batch.append(self.time_select_batch)
                self.ac_data.time_update_clustering.append(self.time_clustering)
            else:
                self.ac_data.time_select_batch.append(time_now - self.start)
                self.ac_data.time_update_clustering.append(time_now - self.start)

    def in_same_cluster(self, o1, o2):
        return self.clustering_solution[o1] == self.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] < 0)

    def initialize_ac_procedure(self):
        self.feedback_freq = np.zeros((self.N, self.N)) + 1
        self.construct_ground_truth_sim_matrix()
        self.construct_initial_sim_matrix()
        self.violations = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i):
                if self.violates_clustering(i, j):
                    self.violations[i, j] = np.abs(self.pairwise_similarities[i, j])
                    self.violations[j, i] = np.abs(self.pairwise_similarities[j, i])

    def construct_ground_truth_sim_matrix(self):
        num_clusters = np.max(self.Y) + 1
        self.num_clusters_ground_truth = num_clusters
        self.ground_truth_clustering = [[] for _ in range(num_clusters)]

        for i in range(len(self.Y)):
            self.ground_truth_clustering[self.Y[i]].append(i)

        self.ground_truth_pairwise_similarities = -1*np.ones((self.N, self.N))
        #self.ground_truth_pairwise_similarities = np.zeros((self.N, self.N))
        for cind in self.ground_truth_clustering:
            self.ground_truth_pairwise_similarities[np.ix_(cind, cind)] = 1

    def clustering_from_clustering_solution(self, clustering_solution):
        num_clusters = np.max(clustering_solution) + 1
        clustering = [[] for _ in range(num_clusters)]
        for i in range(len(clustering_solution)):
            clustering[clustering_solution[i]].append(i)
        return clustering, num_clusters

    def find_cluster(self, d, cluster_indices):
        i = 0
        for ds in cluster_indices:
            if d in ds:
                return i
            i += 1
        print("FIND CLUSTER UNREACHABLE")
        return -1

    def clustering_solution_from_clustering(self, clustering):
        clustering_solution = np.zeros(self.N, dtype=np.uint32)
        for u in range(self.N):
            clustering_solution[u] = self.find_cluster(u, clustering)
        return clustering_solution

    def sim_matrix_from_clustering(self, clustering):
        pairwise_similarities = -self.sim_init*np.ones((self.N, self.N))
        for cind in clustering:
            pairwise_similarities[np.ix_(cind, cind)] = self.sim_init
        return pairwise_similarities

    def construct_initial_sim_matrix(self):
        if self.acq_fn in ["QECC", "COBRAS", "nCOBRAS"]:
            self.sim_init_type = "random_clustering"

        if self.sim_init_type == "zeros":
            self.pairwise_similarities = np.zeros((self.N, self.N))
            self.num_clusters = 1
        elif self.sim_init_type == "uniform_random":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.num_clusters = 1
        elif self.sim_init_type == "uniform_random2":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.pairwise_similarities[np.where(self.pairwise_similarities >= 0)] = self.sim_init
            self.pairwise_similarities[np.where(self.pairwise_similarities < 0)] = -self.sim_init
            self.num_clusters = 1
        elif self.sim_init_type == "inverse_dist":
            D = distance.cdist(self.X, self.X, 'euclidean')
            sim_matrix = np.max(D) - D + np.min(D)
            self.pairwise_similarities = self.sim_init * (2 * sim_matrix - np.max(sim_matrix) -  np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
            np.fill_diagonal(self.pairwise_similarities, 0.0)
            self.num_clusters = 1
        elif self.sim_init_type == "kmeans":
            self.clustering_solution = np.array(self.initial_clustering_solution)
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        elif self.sim_init_type == "random_clustering":
            objects = np.arange(self.N)
            np.random.shuffle(objects)
            k = self.K_init
            cluster_size = int(len(objects) / k)
            cluster_sizes = [cluster_size for _ in range(k)]
            split = np.split(objects, np.cumsum(cluster_sizes))
            clustering = [clust.tolist() for clust in split]
            if len(split) > k:
                clustering[0] += clustering[-1]
                del clustering[-1]
            i = 0
            self.clustering_solution = np.zeros(self.N, dtype=np.uint32)
            for clust in clustering:
                self.clustering_solution[clust] = i
                i += 1
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        else:
            raise ValueError("Invalid sim init type in construct_initial_sim_matrix(...)")
        np.fill_diagonal(self.pairwise_similarities, 0.0)

        pairs = None
        if self.warm_start > 0:
            total_flips = int(self.n_edges * self.warm_start)
            specific_seed = self.repeat_id + self.seed + 42
            state = np.random.get_state()
            np.random.seed(specific_seed)
            pairs = self.qs.select_batch("unif", total_flips)
            for ind1, ind2 in pairs:
                #if np.random.rand() <= self.noise_level:
                #    noisy_val = np.random.uniform(0.15, 0.5)
                #    if np.random.uniform(-1.0, 1.0) > 0:
                #        query = -noisy_val
                #    else:
                #        query = noisy_val
                #else:
                query = self.ground_truth_pairwise_similarities[ind1, ind2]
                self.pairwise_similarities[ind1, ind2] = query
                self.pairwise_similarities[ind2, ind1] = query
                self.feedback_freq[ind1, ind2] += 1
                self.feedback_freq[ind2, ind1] += 1

        self.update_clustering() 

        if self.acq_fn == "cluster_incon" and self.sim_init_type == "zeros" and self.sim_init > 0:
            self.pairwise_similarities_new = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            if pairs is not None:
                for ind1, ind2 in pairs:
                    self.pairwise_similarities_new[ind1, ind2] = self.ground_truth_pairwise_similarities[ind1, ind2]
                    self.pairwise_similarities_new[ind2, ind1] = self.ground_truth_pairwise_similarities[ind1, ind2]
            self.pairwise_similarities = self.pairwise_similarities_new

        if self.warm_start > 0:
            np.random.set_state(state)

    def update_clustering(self):
        self.clustering_solution, _ = max_correlation_dynamic_K(self.pairwise_similarities, self.num_clusters, 5)
        self.clustering = self.clustering_from_clustering_solution(self.clustering_solution)[0]
        self.num_clusters = len(self.clustering)
    
    def update_similarity(self, ind1, ind2):
        if self.feedback_freq[ind1, ind2] > 1:
            self.num_repeat_queries += 1

        self.feedback_freq[ind1, ind2] += 1
        self.feedback_freq[ind2, ind1] += 1

        feedback_frequency = self.feedback_freq[ind1, ind2]
        similarity = self.pairwise_similarities[ind1, ind2]

        if np.random.rand() <= self.noise_level:
            #if self.regular_noise:
                #query = np.random.uniform(-1.0, 1.0)
            #else:
            noisy_val = np.random.uniform(0.15, 0.5)
            if np.random.uniform(-1.0, 1.0) > 0:
                query = -noisy_val
            else:
                query = noisy_val
        else:
            query = self.ground_truth_pairwise_similarities[ind1, ind2]

        self.pairwise_similarities[ind1, ind2] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)
        self.pairwise_similarities[ind2, ind1] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)

    def get_similarity(self, ind1, ind2):
        if np.random.rand() <= self.noise_level:
            return np.random.uniform(-1.0, 1.0)
        else:
            return self.ground_truth_pairwise_similarities[ind1, ind2]
    
    def infer_similarities(self):
        if self.infer_threshold == -1:
            return

        if self.sparse_sim_matrix and not sparse.issparse(self.pairwise_similarities):
            S = sparse.csr_matrix(self.pairwise_similarities)
        else:
            S = self.pairwise_similarities
        clust_sol, q, h = mean_field_clustering(
            S=S, K=self.num_clusters, betas=[self.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.clustering_solution
        )
        
        P_e1_full = np.einsum('ik,jk->ij', q, q)
        distributions = np.stack([P_e1_full, 1 - P_e1_full], axis=-1)

        # Calculating entropy for each distribution
        E_all = scipy_entropy(distributions, base=np.e, axis=-1)

        # Step 2: Compute E_S_ij_minus_1
        E_S_ij_minus_1 = P_e1_full - 1

        # Step 3: Compute E_S_ij
        E_S_ij = (P_e1_full ** 2) + ((1 - P_e1_full) * E_S_ij_minus_1)

        # Step 4: Construct I
        I = E_S_ij  # This is directly E_S_ij as per the provided formula

        # Step 2: Create a condition mask where E_all is smaller than the threshold
        condition_mask = E_all < self.infer_threshold

        # Step 4: Update S based on the combined mask
        self.pairwise_similarities[condition_mask] = I[condition_mask]
        self.feedback_freq[condition_mask] += 1
        np.fill_diagonal(self.pairwise_similarities, 0.0)

        