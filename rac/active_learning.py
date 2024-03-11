import time
import math

from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import itertools
import numpy as np
from scipy.spatial import distance

from rac.query_strategies_AL import QueryStrategyAL
from rac.experiment_data import ExperimentData
from sklearn.neural_network import MLPClassifier

#from torchvision import datasets, transforms
#from torch.utils.data import Dataset, DataLoader

#from rac.utils.utils import CustomDataset
#from rac.utils.models.resnet import ResNet18
from rac.utils.models.vgg import VGG
#from rac.utils.models.simpleNN_net import ThreeLayerNet
#from rac.utils.train_helper import data_train
from rac.correlation_clustering import max_correlation, fast_max_correlation, max_correlation_dynamic_K, mean_field_clustering

from collections import Counter
from collections import defaultdict

from scipy import sparse

#import warnings
#warnings.filterwarnings("once") 

class ActiveLearning:
    def __init__(
            self, 
            repeat_id, 
            X, 
            Y, 
            X_test=None, 
            Y_test=None, 
            transform=None,
            test_transform=None,
            **kwargs
        ):
        self.__dict__.update(kwargs)

        self.X, self.Y = self.random_data_sample(X, Y, self.sample_size)
        self.X_test = X_test
        self.Y_test = Y_test
        self.transform = transform
        self.test_transform = test_transform

        if self.X_test is not None and self.Y_test is not None:
            self.X_test, self.Y_test = self.random_data_sample(X_test, Y_test, self.test_sample_size)

        self.repeat_id = repeat_id
        self.ac_data = ExperimentData(self.Y, repeat_id, **kwargs)
        self.qs = QueryStrategyAL(self)
        self._seed = self.repeat_id+self.seed+317421
        self.N = len(self.Y)
        self.n_edges = (self.N*(self.N-1))/2
        np.random.seed(self._seed)

    def random_data_sample(self, X, Y, size):
        if size <= 1:
            num_samples = int(len(Y)*size)
        else:
            num_samples = np.minimum(size, len(Y))
        inds = np.random.choice(len(Y), num_samples)
        return X[inds], Y[inds]

    def run_AL_procedure(self):
        self.start_time = time.time()
        self.initialize_al_procedure()
        self.store_experiment_data(initial=True)
        stopping_criteria = 25*self.N_pt

        self.ii = 1
        self.num_perfect = 0
        while self.total_queries < stopping_criteria: 
        #while True:
            batch_size = np.minimum(self.batch_size, stopping_criteria - self.total_queries)
            if batch_size != self.batch_size:
                break
            self.start = time.time()
            self.ii += 1
            start_selct_batch = time.time()
            self.selected_indices = self.qs.select_batch(
                acq_fn=self.acq_fn,
                batch_size=batch_size
            )
            self.time_select_batch = time.time() - start_selct_batch

            if len(self.selected_indices) != batch_size:
                raise ValueError("Num queried {} not equal to query size {}!!".format(len(self.selected_indices), batch_size))
            
            self.update_indices()

            time_clustering = time.time()
            self.update_model() 
            self.time_clustering = time.time() - time_clustering

            self.total_queries += batch_size
            self.store_experiment_data()
            self.total_time_elapsed = time.time() - self.start_time

            #num_hours = self.total_time_elapsed / 3600
            #if num_hours > 60:
            #    break

            if self._verbose:
                print("iteration: ", self.ii)
                print("prop_queried: ", self.total_queries/self.N_pt)
                #print("acc: ", accuracy_score(self.Y_pool, self.predictions))
                print("time: ", time.time()-self.start)
                print("num queries: ", len(self.selected_indices))
                print("TIME SELECT BATCH: ", self.time_select_batch)
                print("TIME CLUSTERING: ", self.time_clustering)
                print("X_pool: ", len(self.X_pool))
                print("X_train: ", len(self.X_train))
                print("X_test: ", len(self.X_test))
                print("total queries: ", self.total_queries)
                #print("-----------------")
            
            if accuracy_score(self.Y_pool, self.pool_predictions) == 1.0:
                self.num_perfect += 1
            
            if self.num_perfect >= 5:
                break
                
        return self.ac_data

    def store_experiment_data(self, initial=False):
        self.ac_data.train_accuracy.append(accuracy_score(self.Y_pool[self.queried_indices], self.train_predictions))
        self.ac_data.pool_accuracy.append(accuracy_score(self.Y_pool, self.pool_predictions))
        #self.ac_data.accuracy.append(accuracy_score(self.Y_test, self.test_predictions))
        time_now = time.time() 
        if initial:
            self.ac_data.time_select_batch.append(0.0)
            self.ac_data.time_update_clustering.append(0.0)
            self.ac_data.time.append(time_now - self.start_time)
        else:
            self.ac_data.time.append(time_now - self.start)
            self.ac_data.time_select_batch.append(self.time_select_batch)
            self.ac_data.time_update_clustering.append(self.time_clustering)

    def initialize_al_procedure(self):
        self.N = len(self.Y)
        indices = range(self.N)

        if self.X_test is None:
            self.pool_indices, self.test_indices = train_test_split(
                indices, test_size=0.2, stratify=self.Y, random_state=self._seed
            )
            self.X_test, self.Y_test = self.X[self.test_indices], self.Y[self.test_indices]
            self.X_pool, self.Y_pool = self.X[self.pool_indices], self.Y[self.pool_indices]
        else:
            self.X_pool, self.Y_pool = self.X, self.Y
        self.N_pt = len(self.Y_pool)

        self.initial_train_indices, _ = train_test_split(
            range(len(self.X_pool)), test_size=1-self.warm_start, stratify=self.Y_pool, random_state=self._seed
        )

        self.X_train, self.Y_train = self.X_pool[self.initial_train_indices], self.Y_pool[self.initial_train_indices]
        
        # Convert train_indices_initial to actual labels for stratification in next split
        #init_size = int(self.warm_start * self.N_pt)
        #self.initial_train_indices = np.random.choice(self.N_pt, size=init_size, replace=False)
        self.total_queries = len(self.initial_train_indices)
        self.initial_train_indices = np.array(self.initial_train_indices)
        self.queried_indices = self.initial_train_indices
        self.unqueried_indices = np.setdiff1d(range(self.N_pt), self.queried_indices)

        # labels must be in range {0, ..., n_classes-1} !!!
        self.n_classes = np.max(self.Y) + 1
        self.queried_labels = np.zeros((self.N_pt, self.n_classes))
        self.queried_labels[self.initial_train_indices, self.Y_train] = 1

        # change this @@@@@@@@@@@@@@@@@@@@@@@@
        self.S = np.zeros((self.N_pt, self.N_pt))

        self.Y_pool_queried = np.copy(self.Y_pool)
        self.Y_pool_queried[self.queried_indices] = self.Y_train

        self.update_model()
    
    def update_indices(self):
        for i in self.selected_indices:
            if np.random.rand() < self.noise_level:
                rand_label = np.random.choice(self.n_classes)
                self.queried_labels[i, rand_label] += 1
            else:
                self.queried_labels[i, self.Y_pool[i]] += 1
                
        self.queried_indices = np.where(np.sum(self.queried_labels, axis=1) > 0)[0]
        self.unqueried_indices = np.where(np.sum(self.queried_labels, axis=1) == 0)[0]

        self.X_train = self.X_pool[self.queried_indices]
        self.Y_train = np.argmax(self.queried_labels[self.queried_indices], axis=1)
        self.Y_pool_queried = np.copy(self.Y_pool)
        self.Y_pool_queried[self.queried_indices] = np.copy(self.Y_train)

    def update_model(self):
        if self.model_name == "MLP":
            #self.model = ThreeLayerNet(self.X_train.shape[1], self.n_classes, 100, 100)
            self.model = MLPClassifier(random_state=self._seed, max_iter=500)
            self.model.fit(self.X_train, self.Y_train)
            self.pool_predictions, self.train_predictions = self._predict()
        elif self.model_name == "VGG16":
            self.model = VGG('VGG16')
            #self.train_predictions = dt.get_predictions(dataset)
            #self.test_predictions = dt.get_predictions(test_dataset)
            #self.pool_predictions = dt.get_predictions(pool_dataset)
            #dataset.transform = self.transform
        else:
            pass
        #self._predict()

    def _predict(self):
        if self.predictor == "random":
            predicted_labels = np.random.choice(self.n_classes, size=self.N_pt)
            predicted_labels[self.queried_indices] = self.Y_train
        elif self.predictor == "model":
            predicted_labels = self.model.predict(self.X_pool)
            if self.use_train_pred:
                predicted_labels[self.queried_indices] = self.Y_train
            #dataset = CustomDataset(self.X_train, self.Y_train, transform=self.transform)
            #test_dataset = CustomDataset(self.X_test, self.Y_test, transform=self.test_transform)
            #pool_dataset = CustomDataset(self.X_pool, self.Y_pool_queried, transform=self.test_transform)
            #args = {'n_epoch':150, 'lr':float(0.01), 'batch_size':20, 'max_accuracy':0.99, 'optimizer':'sgd'} 
            #dt = data_train(dataset, self.model, args)
            #self.model = dt.train()
            #self.train_predictions, self.pool_predictions = self._predict()
            ##dataset.transform = self.test_transform
            #dataset = CustomDataset(self.X_train, self.Y_train, transform=self.transform)
            #test_dataset = CustomDataset(self.X_test, self.Y_test, transform=self.test_transform)
            #pool_dataset = CustomDataset(self.X_pool, self.Y_pool_queried, transform=self.test_transform)
            #predicted_labels = self.model.predict(self.X_pool)
            #self.train_predictions = dt.get_predictions(dataset)
            #self.test_predictions = dt.get_predictions(test_dataset)
            #self.pool_predictions = dt.get_predictions(pool_dataset)
            #return self.pool_predictions, self.train_predictions
        elif self.predictor == "CC":
            max_indices = np.argmax(self.queried_labels, axis=1)
            prob_all = np.zeros(self.queried_labels.shape)
            prob_all[np.arange(len(max_indices)), max_indices] = 1
            #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
            #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

            # Predict probabilities for X_pool
            prob_pool = self.model.predict_proba(self.X_pool)

            prob_all[self.unqueried_indices] = prob_pool[self.unqueried_indices]
            if not self.use_train_pred:
                prob_all = prob_pool

            # Initialize similarity matrix
            N = self.N_pt
            for i in range(N):
                for j in range(0, i):
                    if self.sim_init == "t1":
                        if i in self.queried_indices and j in self.queried_indices:
                            self.S[i, j] = 1 if self.Y_pool_queried[i] == self.Y_pool_queried[j] else -1
                            self.S[j, i] = self.S[i, j]
                        else:
                            self.S[i, j] = 0
                            self.S[j, i] = 0
                    else:
                        P_S_ij_plus_1 = np.sum(prob_all[i, :] * prob_all[j, :])
                        E_S_ij_plus_1 = P_S_ij_plus_1
                        E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                        E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                        self.S[i, j] = E_S_ij
                        self.S[j, i] = self.S[i, j]

            # Ensure diagonal is zero
            np.fill_diagonal(self.S, 0)

            self.clustering_solution, _ = max_correlation(self.S, self.n_classes, 5)
            #clust_sol, q, h = mean_field_clustering(
            #    S=self.S, K=self.n_classes, betas=[self.mean_field_beta], max_iter=150, tol=1e-10, 
            #    predicted_labels=self.clustering_solution
            #)

            predicted_labels = np.array([None]*len(self.Y_pool))  # Initialize all predictions as None
            predicted_labels[self.queried_indices] = np.copy(self.Y_train)

            cluster_labels = {}
            for cluster in np.unique(self.clustering_solution):
                indices_in_cluster = np.where(self.clustering_solution == cluster)[0]
                labeled_indices_in_cluster = np.intersect1d(indices_in_cluster, self.queried_indices)
                
                if labeled_indices_in_cluster.size > 0:
                    labels_in_cluster = self.Y_pool_queried[labeled_indices_in_cluster]
                    most_common_label, _ = Counter(labels_in_cluster).most_common(1)[0]
                    cluster_labels[cluster] = most_common_label
                else:
                    cluster_labels[cluster] = None  # Mark for special handling

            # For each unlabeled data point, either use the cluster's common label or compute similarity
            for i, label in enumerate(predicted_labels):
                if label is None:  # Unlabeled data point
                    cluster = self.clustering_solution[i]
                    if cluster_labels[cluster] is not None:
                        # Use the most common label if the cluster has labeled data
                        predicted_labels[i] = cluster_labels[cluster]
                    else:
                        # Compute summed similarity for each class and assign the class with the highest summed similarity
                        class_similarities = defaultdict(float)
                        for idx in self.queried_indices:
                            class_similarities[self.Y_pool_queried[idx]] += self.S[i, idx]
                        
                        # Assign the class with the highest total similarity if there are any labeled points to compare with
                        if class_similarities:
                            predicted_labels[i] = max(class_similarities, key=class_similarities.get)
                        # Optional: Handle the case with no reference labeled points in a special manner, e.g., assign a default label
        elif self.predictor == "CC2":
            max_indices = np.argmax(self.queried_labels, axis=1)
            prob_all = np.zeros(self.queried_labels.shape)
            prob_all[np.arange(len(max_indices)), max_indices] = 1
            #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
            #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

            # Predict probabilities for X_pool
            prob_pool = self.model.predict_proba(self.X_pool)

            prob_all[self.unqueried_indices] = prob_pool[self.unqueried_indices]

            N = prob_all.shape[0]
            for i in range(N):
                for j in range(0, i):
                    if i != j:
                        if self.sim_init == "t1":
                            if i in self.queried_indices and j in self.queried_indices:
                                self.S[i, j] = 1 if self.Y_pool_queried[i] == self.Y_pool_queried[j] else -1
                                self.S[j, i] = self.S[i, j]
                            else:
                                self.S[i, j] = 0
                                self.S[j, i] = 0
                        else:
                            P_S_ij_plus_1 = np.sum(prob_all[i, :] * prob_all[j, :])
                            E_S_ij_plus_1 = P_S_ij_plus_1
                            E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                            E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                            self.S[i, j] = E_S_ij
                            self.S[j, i] = self.S[i, j]

            # Ensure diagonal is zero
            np.fill_diagonal(self.S, 0)

            self.clustering_solution, _ = max_correlation(self.S, self.n_classes, 5)
            clust_sol, q, h = mean_field_clustering(
                S=self.S, K=self.n_classes, betas=[self.mean_field_beta], max_iter=150, tol=1e-10, 
                predicted_labels=self.clustering_solution
            )

            # Ensure diagonal is zero
            np.fill_diagonal(self.S, 0)
            predicted_labels = np.array([None]*len(self.Y_pool))
            # Initialize predicted labels for labeled points
            predicted_labels[self.queried_indices] = self.Y_train
            
            # Map each class to its labeled indices
            class_to_indices = defaultdict(list)
            for index in self.queried_indices:
                class_to_indices[self.Y_pool_queried[index]].append(index)

            for i in range(len(self.Y_pool)):
                if predicted_labels[i] is None:  # If unlabeled
                    # Compute total similarity to all labeled objects of each class
                    class_similarities = {}
                    for class_label, indices in class_to_indices.items():
                        # Sum of similarities between i and all labeled objects of class_label
                        class_similarities[class_label] = self.S[i, indices].sum()

                    # Assign the class with the highest total similarity
                    if class_similarities:  # Check if we have any class similarities computed
                        predicted_labels[i] = max(class_similarities, key=class_similarities.get)
                    else:
                        raise ValueError("No class similarities computed for data point {}!")
                        # Handle the case where there might be no labeled data at all to compare with
                        # This could be set to a default value or handled in another specific way
        pool_predictions = predicted_labels.astype(np.int32)
        train_predictions = predicted_labels[self.queried_indices].astype(np.int32)
        return pool_predictions, train_predictions
            


        


