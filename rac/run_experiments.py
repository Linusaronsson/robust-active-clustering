import os
import multiprocessing as mp
import argparse
import json
#import time
import pickle

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification

import itertools
from rac.active_clustering import ActiveClustering
from pathlib import Path

def get_dataset(**options):
    dataset = options["dataset"]
    seed = options["seed"]
    normalize = False
    if dataset == "synthetic":
        class_balance = options["class_balance"]
        n_clusters = options["n_clusters"]
        n_samples = options["n_samples"]
        if class_balance == None:
            weights = None
        else:
            prop = (1-class_balance)/(n_clusters-1)
            weights = [class_balance]
            weights += [prop]*(n_clusters-1)
        X, Y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_clusters,
            n_clusters_per_class=1,
            weights=weights,
            flip_y=0,
            class_sep=1.8,
            hypercube=True,
            shift=0.0,
            scale=1.0,
            shuffle=True,
            random_state=seed)
        normalize = False
    elif dataset == "20newsgroups":
        X = np.load("datasets/20newsgroups_data/X.npy")
        Y = np.load("datasets/20newsgroups_data/Y.npy")
    elif dataset == "cifar10":
        X = np.load("datasets/cifar10_data/X.npy")
        Y = np.load("datasets/cifar10_data/Y.npy")
    elif dataset == "cifar10_original":
        X = np.load("datasets/cifar10_original_data/X.npy")
        Y = np.load("datasets/cifar10_original_data/Y.npy")
    elif dataset == "mnist":
        X = np.load("datasets/mnist_data/X.npy")
        Y = np.load("datasets/mnist_data/Y.npy")
    elif dataset == "breast_cancer":
        X = np.load("datasets/breast_cancer_data/X.npy")
        Y = np.load("datasets/breast_cancer_data/Y.npy")
    elif dataset == "cardiotocography":
        X = np.load("datasets/cardiotocography_data/X.npy")
        Y = np.load("datasets/cardiotocography_data/Y.npy")
    elif dataset == "ecoli":
        X = np.load("datasets/ecoli_data/X.npy")
        Y = np.load("datasets/ecoli_data/Y.npy")
    elif dataset == "forest_type_mapping":
        X = np.load("datasets/ForestTypeMapping_data/X.npy")
        Y = np.load("datasets/ForestTypeMapping_data/Y.npy")
    elif dataset == "mushrooms":
        X = np.load("datasets/mushrooms_data/X.npy")
        Y = np.load("datasets/mushrooms_data/Y.npy")
    elif dataset == "user_knowledge":
        X = np.load("datasets/user_knowledge_data/X.npy")
        Y = np.load("datasets/user_knowledge_data/Y.npy")
    elif dataset == "yeast":
        X = np.load("datasets/yeast_data/X.npy")
        Y = np.load("datasets/yeast_data/Y.npy")
    elif dataset == "20newsgroups_small": ##############################
        X = np.load("datasets_small/20newsgroups_data/X.npy")
        Y = np.load("datasets_small/20newsgroups_data/Y.npy")
    elif dataset == "cifar10_small":
        X = np.load("datasets_small/cifar10_data/X.npy")
        Y = np.load("datasets_small/cifar10_data/Y.npy")
    elif dataset == "mnist_small":
        X = np.load("datasets_small/mnist_data/X.npy")
        Y = np.load("datasets_small/mnist_data/Y.npy")
    elif dataset == "breast_cancer_small":
        X = np.load("datasets_small/breast_cancer_data/X.npy")
        Y = np.load("datasets_small/breast_cancer_data/Y.npy")
    elif dataset == "cardiotocography_small":
        X = np.load("datasets_small/cardiotocography_data/X.npy")
        Y = np.load("datasets_small/cardiotocography_data/Y.npy")
    elif dataset == "ecoli_small":
        X = np.load("datasets_small/ecoli_data/X.npy")
        Y = np.load("datasets_small/ecoli_data/Y.npy")
    elif dataset == "forest_type_mapping_small":
        X = np.load("datasets_small/ForestTypeMapping_data/X.npy")
        Y = np.load("datasets_small/ForestTypeMapping_data/Y.npy")
    elif dataset == "mushrooms_small":
        X = np.load("datasets_small/mushrooms_data/X.npy")
        Y = np.load("datasets_small/mushrooms_data/Y.npy")
    elif dataset == "user_knowledge_small":
        X = np.load("datasets_small/user_knowledge_data/X.npy")
        Y = np.load("datasets_small/user_knowledge_data/Y.npy")
    elif dataset == "yeast_small":
        X = np.load("datasets_small/yeast_data/X.npy")
        Y = np.load("datasets_small/yeast_data/Y.npy")
    else:
        raise ValueError("INVALID DATASET")
    if normalize:
        X = preprocessing.StandardScaler().fit_transform(X)
        #X = preprocessing.MinMaxScaler().fit_transform(X)
    return X, Y

def gather_results(result_queue, general_options):
    try:
        while True:
            ac_data = result_queue.get(block=True, timeout=None)
            if ac_data is None:
                return
            path = general_options["path"]
            experiment_path = path + ac_data.dataset_name
            data_path = experiment_path + "/" + ac_data.hashed_name
            if not os.path.exists(data_path):
                Path(experiment_path).mkdir(parents=True, exist_ok=True)
                if os.path.exists(experiment_path):
                    with open(data_path, 'wb') as handle:
                        pickle.dump([ac_data], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(data_path, 'rb') as handle:
                    exp = pickle.load(handle)
                exp.append(ac_data)
                with open(data_path, 'wb') as handle:
                    pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)

            completed_experiments_path = path + "completed_experiments.txt"
            with open(completed_experiments_path, "a") as file:
                file.write(ac_data.name_repeat + "\n")

    except EOFError as e:
        print("EOF ERROR GATHER RESULTS !! !! @ ")
        print(e)
        print("--------------")

def run_experiment(experiment_queue, result_queue, worker_id, general_options):
    try:
        while True:
            ac = experiment_queue.get(block=True, timeout=None)
            if ac is None:
                return
            ac_data = ac.ac_data
            path = general_options["path"]
            print("#### Worker ID: {} ####".format(worker_id))
            already_completed = False
            completed_experiments_path = path + "completed_experiments.txt"
            with open(completed_experiments_path) as file:
                for line in file:
                    if line.strip() == ac_data.name_repeat:
                        already_completed = True
                        break
            if already_completed:
                print(ac.ac_data.name_repeat + " already completed")
                continue
            else:
                print(ac.ac_data.name_repeat + " running")
            print("#### ####")
            ac_data = ac.run_AL_procedure()
            result_queue.put(ac_data)
    except EOFError as e:
        print("EOF ERROR RUN_EXPERIMENT !! !! !! !! !!")
        print(e)
        print("--------------")
        return

def get_keys_from_options(config):
    options_keys = []
    options_values = []
    for key1, value1 in config.items():
        for key2, value2 in value1.items():
            if type(value2) != list:
                value2 = [value2]
            options_keys.append(key2)
            options_values.append(value2)
    return options_keys, options_values

def run_experiments(config):
    manager = mp.Manager()
    experiment_queue = manager.Queue() 
    result_queue = manager.Queue() 
    general_options = config["general_options"]
    all_options = {}
    for option_category, options in config.items():
        if option_category not in all_options:
            all_options[option_category] = list(options.keys())

    options_keys, options_values = get_keys_from_options(config)
    saved_datasets = {}
    for repeat_id in range(general_options["num_repeats"]):
        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))

            k = exp_kwargs["K_init"]
            dataset_name = str(k)
            for dataset_key in all_options["dataset_options"]:
                dataset_name += str(exp_kwargs[dataset_key])

            seed = exp_kwargs["seed"]
            if dataset_name not in saved_datasets:
                saved_datasets[dataset_name] = {}
                X, Y = get_dataset(**exp_kwargs)
                saved_datasets[dataset_name]["X"] = X
                saved_datasets[dataset_name]["Y"] = Y
                if exp_kwargs["sim_init_type"] == "custom":
                    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X)
                    saved_datasets[dataset_name]["initial_labels"] = kmeans.labels_
                else:
                    saved_datasets[dataset_name]["initial_labels"] = None

            initial_labels = saved_datasets[dataset_name]["initial_labels"]
            X = saved_datasets[dataset_name]["X"]
            Y = saved_datasets[dataset_name]["Y"]

            kwargs = {}
            for option_category, options in all_options.items():
                if option_category not in kwargs:
                    kwargs[option_category] = {}
                for option in options:
                    kwargs[option_category][option] = exp_kwargs[option]

            ac = ActiveClustering(X, Y, repeat_id, initial_labels, **kwargs)
            experiment_queue.put(ac)

    if not general_options["local"]:
        general_options["path"] = "/mimer/NOBACKUP/groups/active-learning/experiment_results/" 
    else:
        general_options["path"] = "experiment_results_local/" 
        
    general_options["path"] += general_options["experiment_name"] + "/"
    path = general_options["path"]
    exp_results = Path(path)
    completed_exps = Path(path + "completed_experiments.txt")
    exp_results.mkdir(parents=True, exist_ok=True)
    completed_exps.touch(exist_ok=True)

    processes = []
    for worker in range(general_options["n_workers"]):
        process = mp.Process(target=run_experiment, args=(experiment_queue, result_queue, worker, general_options), daemon=False)
        process.start()
        processes.append(process)
        experiment_queue.put(None)
    gather_process = mp.Process(target=gather_results, args=(result_queue, general_options), daemon=False)
    gather_process.start()
    for process in processes:
        process.join()
    result_queue.put(None) 
    gather_process.join()

def read_config_file(filename):
    if filename.split('.')[-1] not in ['json']:
        raise IOError('Only json type are supported now!')
	
    if not os.path.exists(filename):
        raise FileNotFoundError('Config file does not exist!')
        
    with open(filename, 'r') as f:
        config = json.load(f)
    return config	

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, help="Config")

    args = parser.parse_args()
    mp.set_start_method("spawn")

    if ".json" in args.config:
        config = read_config_file(args.config)
        run_experiments(config)
    else:
        for file in os.listdir(args.config):
            filename = os.fsdecode(file)
            config = read_config_file(args.config + "/" + filename)
            run_experiments(config)

if __name__ == "__main__":
    main()