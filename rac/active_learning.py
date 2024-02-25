import time
import math

from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
import itertools
import numpy as np
from scipy.spatial import distance

from rac.query_strategies_AL import QueryStrategyAL
from rac.experiment_data import ExperimentData
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

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
        stopping_criteria = self.N_pt

        ii = 1
        while self.total_queries < stopping_criteria: 
        #while True:
            batch_size = np.minimum(self.batch_size, stopping_criteria - self.total_queries)
            self.start = time.time()
            ii += 1
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
                print("iteration: ", ii)
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
        train_indices_initial, self.test_indices = train_test_split(indices, test_size=0.2, random_state=self._seed)
        self.train_indices, self.pool_indices = train_test_split(
            train_indices_initial,
            test_size=1-self.warm_start,
            random_state=self._seed
        )
        self.train_indices = np.array(self.train_indices)
        self.test_indices = np.array(self.test_indices)
        self.pool_indices = np.array(self.pool_indices)
        self.X_train, self.Y_train = self.X[self.train_indices], self.Y[self.train_indices]
        self.X_test, self.Y_test = self.X[self.test_indices], self.Y[self.test_indices]
        self.X_pool, self.Y_pool = self.X[self.pool_indices], self.Y[self.pool_indices]
        self.N_pt = len(self.Y_pool) + len(self.Y_train)
        self.total_queries = len(self.Y_train)
        self.initialize_model()
        self.update_model()

    def initialize_model(self):
        if self.model_name == "GP":
            kernel = 1.0 * RBF(1.0)
            self.model = GaussianProcessClassifier(kernel=kernel, random_state=self._seed, n_restarts_optimizer=10)
        else:
            raise ValueError("Invalid model name: {}".format(self.model_name))
    
    def update_indices(self):
        self.pool_indices = np.setdiff1d(self.pool_indices, self.selected_indices)
        self.train_indices = np.concatenate((self.train_indices, self.selected_indices))
        self.X_pool, self.Y_pool = self.X[self.pool_indices], self.Y[self.pool_indices]
        self.X_train, self.Y_train = self.X[self.train_indices], self.Y[self.train_indices]

    def update_model(self):
        if self.model_name == "GP":
            kernel = 1.0 * RBF(1.0)
            self.model = GaussianProcessClassifier(kernel=kernel, random_state=self._seed)
            gpc = self.model.fit(self.X_train, self.Y_train)
            self.predictions = gpc.predict(self.X_test)
            #self.pred_probs = gpc.predict_proba(self.X_test)
        else:
            pass
        


