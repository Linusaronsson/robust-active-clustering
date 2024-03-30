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

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from rac.utils.utils import LabeledToUnlabeledDataset, CustomDataset
from rac.utils.models.resnet import ResNet18
from rac.utils.models.vgg import VGG
from rac.utils.models.simpleNN_net import ThreeLayerNet
from rac.utils.train_helper import data_train
from rac.correlation_clustering import max_correlation, fast_max_correlation, max_correlation_dynamic_K, mean_field_clustering

from rac.active_learning_strategies.strategy import Strategy

from collections import Counter
from collections import defaultdict

from scipy.stats import entropy as scipy_entropy
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
        rs = np.random.RandomState(13)
        if size <= 1:
            num_samples = int(len(Y)*size)
        else:
            num_samples = np.minimum(size, len(Y))
        inds = rs.choice(len(Y), num_samples)
        return X[inds], Y[inds]

    def run_AL_procedure(self):
        self.start_time = time.time()
        self.initialize_al_procedure()
        self.store_experiment_data(initial=True)
        stopping_criteria = self.n_passes*self.N_pt

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
            
            #if accuracy_score(self.Y_test, self.test_predictions) == 1.0:
            #    self.num_perfect += 1
            
            #if self.num_perfect >= 5:
            #    break
                
        return self.ac_data

    def store_experiment_data(self, initial=False):
        self.ac_data.pool_accuracy.append(accuracy_score(self.Y_pool, self.pool_predictions))
        self.ac_data.accuracy.append(accuracy_score(self.Y_test, self.test_predictions))
        self.ac_data.train_accuracy.append(accuracy_score(self.Y_pool[self.queried_indices], self.train_predictions))

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
                indices, test_size=0.2, stratify=self.Y, random_state=13
            )
            self.X_test, self.Y_test = self.X[self.test_indices], self.Y[self.test_indices]
            self.X_pool, self.Y_pool = self.X[self.pool_indices], self.Y[self.pool_indices]
        else:
            self.X_pool, self.Y_pool = self.X, self.Y
        self.N_pt = len(self.Y_pool)

        self.initial_train_indices, _ = train_test_split(
            range(len(self.X_pool)), test_size=1-self.warm_start, stratify=self.Y_pool, random_state=14
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

        self.wrong_labels = np.zeros((self.N_pt, self.n_classes))

        # change this @@@@@@@@@@@@@@@@@@@@@@@@
        self.N_total = self.N_pt + len(self.X_test)
        self.S = np.zeros((self.N_total, self.N_total))

        self.Y_pool_queried = np.copy(self.Y_pool)
        self.Y_pool_queried[self.queried_indices] = self.Y_train

        self.update_model()
    
    
    def get_pred(self, i):
        probs_sorted = np.argsort(-self.probs[i])
        for p_arg in probs_sorted:
            if self.wrong_labels[i, p_arg] == 0:
                return p_arg
        return probs_sorted[0]

    def update_indices(self):
        classes = np.unique(self.Y_pool)
        if self.oracle == "soft":
            for i in self.selected_indices:
                if np.random.rand() < self.noise_level:
                    true_label = np.random.choice(self.n_classes)
                else:
                    true_label = self.Y_pool[i]

                pred_label = self.get_pred(i)

                if pred_label == true_label:
                    self.queried_labels[i, true_label] += 1
                    remaining_classes = np.setdiff1d(classes, true_label)
                    self.wrong_labels[i, remaining_classes] += 1
                else:
                    self.wrong_labels[i, pred_label] += 1
        else:
            for i in self.selected_indices:
                if np.random.rand() < self.noise_level:
                    true_label = np.random.choice(self.n_classes)
                else:
                    true_label = self.Y_pool[i]

                self.queried_labels[i, true_label] += 1

        self.queried_indices = np.where(np.sum(self.queried_labels, axis=1) > 0)[0]
        self.unqueried_indices = np.where(np.sum(self.queried_labels, axis=1) == 0)[0]

        self.X_train = self.X_pool[self.queried_indices]
        self.Y_train = np.argmax(self.queried_labels[self.queried_indices], axis=1)
        self.Y_pool_queried = np.copy(self.Y_pool)
        self.Y_pool_queried[self.queried_indices] = np.copy(self.Y_train)

    def update_model(self):
        if self.model_name == "MLP":
            self.model = ThreeLayerNet(self.X_train.shape[1], self.n_classes, 100, 100)
        elif self.model_name == "VGG16":
            self.model = VGG('VGG16', channels=1)
        elif self.model_name == "resnet":
            if self.dataset_name == "mnist_original":
                channels = 1
            else:
                channels = 3
            self.model = ResNet18(num_classes=self.n_classes, channels=channels)
        else:
            pass

        args = {'n_epoch':100, 'lr':float(0.001), 'batch_size':20, 'max_accuracy':1.1, 'optimizer':'adam'} 
        train_dataset = CustomDataset(self.X_train, self.Y_train, transform=self.transform)
        dt = data_train(train_dataset, self.model, args)
        clf = dt.train()
        pool_dataset = CustomDataset(self.X_pool, self.Y_pool_queried, transform=self.test_transform)
        es = Strategy(None, None, self.model, self.n_classes)
        self.probs = es.predict_prob(LabeledToUnlabeledDataset(pool_dataset)).cpu().numpy()
        if self.oracle == "soft":
            self.probs = self.renormalize_softmax(self.probs)
        test_dataset = CustomDataset(self.X_test, self.Y_test, transform=self.test_transform)
        self.probs_test = es.predict_prob(LabeledToUnlabeledDataset(test_dataset)).cpu().numpy()

        if self.predictor == "CC" or self.acq_fn == "cc_entropy":
            self.construct_sims()

        self.pool_predictions, self.train_predictions, self.test_predictions = self._predict()

    def construct_sims(self):
        # Initialize similarity matrix

        probs = np.concatenate((self.probs, self.probs_test), axis=0)

        for i in range(self.N_total):
            for j in range(0, i):
                P_S_ij_plus_1 = np.sum(probs[i, :] * probs[j, :])
                E_S_ij_plus_1 = P_S_ij_plus_1
                E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                self.S[i, j] = E_S_ij
                self.S[j, i] = self.S[i, j]
                if np.random.rand() < self.sim_noise_level:
                    self.S[i, j] = np.random.uniform(-0.001, 0.001)
                    self.S[j, i] = self.S[i, j]


        # Ensure diagonal is zero
        np.fill_diagonal(self.S, 0)

    def renormalize_softmax(self, prob):
        N_pt, n_classes = prob.shape
        adjusted_prob = np.copy(prob).astype(float)
        
        for i in range(N_pt):
            for j in range(n_classes):
                if self.queried_labels[i, j] > 0 and self.wrong_labels[i, j] > 0:
                    # Handle conflict by considering the proportion between queried and wrong labels
                    proportion = self.queried_labels[i, j] / self.wrong_labels[i, j]
                    adjusted_prob[i, j] *= proportion
                elif self.queried_labels[i, j] > 0:
                    # If only queried labels are present, boost the probability
                    adjusted_prob[i, j] *= (30 + self.queried_labels[i, j])
                elif self.wrong_labels[i, j] > 0:
                    # If only wrong labels are present, reduce the probability
                    adjusted_prob[i, j] *= (1 / (30 + self.wrong_labels[i, j]))
            
            # Normalize to ensure the probabilities sum to 1
            adjusted_prob[i, :] /= np.sum(adjusted_prob[i, :])
        
        return adjusted_prob
                
    def _predict(self):
        if self.predictor == "model":
            predicted_labels = np.argmax(self.probs, axis=1)
            pool_predictions = predicted_labels.astype(np.int32)
            train_predictions = predicted_labels[self.queried_indices].astype(np.int32)
            test_predictions = np.argmax(self.probs_test, axis=1).astype(np.int32)
            return pool_predictions, train_predictions, test_predictions
        elif self.predictor == "CC":
            self.clustering_solution, _ = fast_max_correlation(self.S, self.n_classes, 3)
            #clust_sol, q, h = mean_field_clustering(
            #    S=self.S, K=self.n_classes, betas=[self.mean_field_beta], max_iter=150, tol=1e-10, 
            #    predicted_labels=self.clustering_solution
            #)

            predicted_labels = np.zeros(self.N_total)  # Initialize all predictions as None
            #predicted_labels[self.queried_indices] = np.copy(self.Y_train)

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

            pool_predictions = predicted_labels[:self.N_pt].astype(np.int32)
            train_predictions = predicted_labels[self.queried_indices].astype(np.int32)
            test_predictions = predicted_labels[self.N_pt:].astype(np.int32)
            return pool_predictions, train_predictions, test_predictions
            


        


