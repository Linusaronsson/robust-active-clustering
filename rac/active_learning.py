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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

from rac.utils.models.resnet import ResNet18
from rac.utils.models.vgg import VGG

from scipy import sparse

#import warnings
#warnings.filterwarnings("once") 

class ActiveLearning:
    def __init__(self, X, Y, repeat_id, **kwargs):
        self.__dict__.update(kwargs)

        self.X, self.Y = X, Y
        self.repeat_id = repeat_id
        self.ac_data = ExperimentData(Y, repeat_id, **kwargs)
        self.qs = QueryStrategyAL(self)
        self._seed = self.repeat_id+self.seed+317421
        self.N = len(self.Y)
        self.n_edges = (self.N*(self.N-1))/2
        np.random.seed(self._seed)

    def run_AL_procedure(self):
        self.start_time = time.time()
        self.initialize_al_procedure()
        self.store_experiment_data(initial=True)
        stopping_criteria = 2*self.N_pt

        self.ii = 1
        while self.total_queries < stopping_criteria: 
        #while True:
            batch_size = np.minimum(self.batch_size, stopping_criteria - self.total_queries)
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
                print("acc: ", accuracy_score(self.Y_test, self.predictions))
                print("time: ", time.time()-self.start)
                print("num queries: ", len(self.selected_indices))
                print("TIME SELECT BATCH: ", self.time_select_batch)
                print("TIME CLUSTERING: ", self.time_clustering)
                print("X_pool: ", len(self.X_pool))
                print("X_train: ", len(self.X_train))
                print("X_test: ", len(self.X_test))
                print("total queries: ", self.total_queries)
                #print("-----------------")
                
        return self.ac_data

    def store_experiment_data(self, initial=False):
        self.ac_data.accuracy.append(accuracy_score(self.Y_test, self.predictions))
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
        self.pool_indices, self.test_indices = train_test_split(
            indices, test_size=0.2, stratify=self.Y, random_state=self._seed
        )
        self.N_pt = len(self.pool_indices)
        self.X_test, self.Y_test = self.X[self.test_indices], self.Y[self.test_indices]
        self.X_pool, self.Y_pool = self.X[self.pool_indices], self.Y[self.pool_indices]

        self.initial_train_indices, _ = train_test_split(
            range(len(self.pool_indices)), test_size=1-self.warm_start, stratify=self.Y_pool, random_state=self._seed
        )

        self.X_train, self.Y_train = self.X_pool[self.initial_train_indices], self.Y_pool[self.initial_train_indices]
        
        # Convert train_indices_initial to actual labels for stratification in next split
        #init_size = int(self.warm_start * self.N_pt)
        #self.initial_train_indices = np.random.choice(self.N_pt, size=init_size, replace=False)
        self.total_queries = len(self.initial_train_indices)
        self.initial_train_indices = np.array(self.initial_train_indices)

        self.n_classes = np.max(self.Y) + 1
        self.queried_labels = np.zeros((self.N_pt, self.n_classes))
        self.queried_labels[self.initial_train_indices, self.Y_train] = 1
        self.update_model()
    
    def update_indices(self):
        for i in self.selected_indices:
            if np.random.rand() < self.noise_level:
                rand_label = np.random.choice(self.n_classes)
                self.queried_labels[i, rand_label] += 1
            else:
                self.queried_labels[i, self.Y_pool[i]] += 1
                
        queried_indices = np.where(np.sum(self.queried_labels, axis=1) > 0)[0]
        self.X_train = self.X_pool[queried_indices]
        self.Y_train = np.argmax(self.queried_labels[queried_indices], axis=1)

    def update_model(self):
        if self.model_name == "GP":
            kernel = 1.0 * RBF(1.0)
            self.model = GaussianProcessClassifier(kernel=kernel, random_state=self._seed)
            gpc = self.model.fit(self.X_train, self.Y_train)
            self.predictions = gpc.predict(self.X_test)
            #self.pred_probs = gpc.predict_proba(self.X_test)
        elif self.model_name == "MLP":
            self.model = MLPClassifier(random_state=self._seed, max_iter=500)
            gpc = self.model.fit(self.X_train, self.Y_train)
            self.predictions = gpc.predict(self.X_test)
        elif self.model_name == "VGG16":
            self.model = VGG('VGG16')
            pass
        else:
            pass
        


