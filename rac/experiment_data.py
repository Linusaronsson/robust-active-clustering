import random
import time
import os
import pickle
import math

import numpy as np
import itertools
from hashlib import sha256
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import copy

class ExperimentData:
    def __init__(self, Y, repeat_id, **kwargs):
        for key, value in kwargs.items():
            self.__dict__.update(kwargs[key])


        self.repeat_id = repeat_id
        self.experiment_params = []
        #self.name = str(repeat_id) + "_"
        self.name = ""
        for key, value in kwargs.items():
            if key == "general_options":
                continue
            for key, value in value.items():
                self.name += str(value) + "_"
                self.experiment_params.append(key)
        self.name = self.name[:-1]
        self.name_repeat = self.name + "_" + str(repeat_id)

        self.hashed_name = sha256(self.name.encode("utf-8")).hexdigest()

        self.dataset_name = ""
        for key, value in kwargs["dataset_options"].items():
            self.dataset_name += str(value) + "_"
        self.dataset_name = self.dataset_name[:-1]

        # ground truth clustering solution
        self.Y = Y

        # num queries and estimated pw sims at last iteration
        self.feedback_freq = None
        self.pairwise_similarities = None

        # info about current pairwise similarities
        self.num_pos = []
        self.num_neg = []
        self.num_neg_ground_truth = []
        self.num_pos_ground_truth = []

        self.accuracy = []
        self.precision = []
        self.recall = []
        self.precision_neg = []
        self.recall_neg = []
        self.f1_score = []
        self.f1_score_weighted = []

        self.accuracy_clustering = []
        self.precision_clustering = []
        self.recall_clustering = []
        self.precision_neg_clustering = []
        self.recall_neg_clustering = []
        self.f1_score_clustering = []
        self.f1_score_weighted_clustering = []

        # info about clustering
        self.rand = []
        self.ami = []
        self.v_measure = []
        self.num_clusters = []

        # info about bad triangles
        self.num_violations = []

        # other useful information
        self.num_queries = []
        self.time = []

        # mnist stuff
        self.avg_cluster_images = []

    def is_equal_no_repeat(self, a2):
        params = set(self.experiment_params + a2.experiment_params)
        for param in params:
            if not hasattr(self, param) or not hasattr(a2, param):
                continue
            if getattr(self, param) != getattr(a2, param):
                return False
        return True

    def is_equal(self, a2):
        return self.is_equal_no_repeat(a2) and self.repeat_id == a2.repeat_id

class ExperimentReader:
    def __init__(self, metrics=None):
        self.metrics = metrics
        if self.metrics is not None:
            self.metrics.append("rand")
            self.metrics = list(set(self.metrics))
        self.Y = None
        self.num_feedback = None

    def get_metric_data(self, exp):
        data = {}
        for metric in self.metrics:
            for e in exp:
                if metric not in data:
                    data[metric] = [getattr(e, metric)]
                else:
                    data[metric].append(getattr(e, metric))
        return data

    def extend_list(self, data, max_size):
        if len(data) < max_size:
            data.extend([data[-1]] * (max_size - len(data)))
        #if len(data) > max_size:                
            #raise ValueError("Length of data larger than max size")
        return data

    def read_all_data(self, folder):
        data = pd.DataFrame()
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name == "completed_experiments.txt":
                    continue
                with open(path + "/" + name, 'rb') as handle:
                    try:
                        exp = pickle.load(handle)
                    except EOFError:
                        print("EOF error in read data")
                        exp = []
                    
                    dat = exp[0].__dict__
                    if self.Y is None:
                        self.Y = exp[0].Y
                        self.dataset = exp[0].dataset_name
                        self.num_feedback = exp[0].num_feedback
                    wanted_keys = exp[0].experiment_params
                    sub_dat = dict((k, dat[k]) for k in wanted_keys if k in dat)
                    sub_dat
                    #sub_dat["data"] = exp
                    metric_data = self.get_metric_data(exp)
                    for metric in self.metrics:
                        met_data = metric_data[metric]
                        max_size = 0
                        for mt in met_data:
                            if len(mt) > max_size:
                                max_size = len(mt)
                        for i in range(len(met_data)):
                            met_data[i] = self.extend_list(met_data[i], max_size)
                        sub_dat[metric] = np.array(met_data)
                    sub_dat = pd.DataFrame([sub_dat])
                    data = pd.concat([data, sub_dat], ignore_index=True)
        if len(data) == 0:
            print("no data found in folder")
            return None
        return data

    def flatten_dataframe(self, df, non_data_column_names, data_column_names):
        # List to store each DataFrame
        dataframes = []
        for data_column in data_column_names:
            # Temporarily drop non-required columns
            df_temp = df.drop(columns=[col for col in data_column_names if col != data_column])
            
            # Expand each array into its own DataFrame and merge back with the original DataFrame
            for i in range(df_temp.shape[0]):
                array = df_temp.loc[i, data_column]
                df_expanded = pd.DataFrame(array).unstack().reset_index()
                df_expanded.columns = ['x', 'x2', 'y']
                for col in non_data_column_names:
                    df_expanded[col] = [df_temp.loc[i, col]] * len(df_expanded)
                df_expanded['metric'] = data_column
                dataframes.append(df_expanded)

        # Concatenate all dataframes
        df_flattened = pd.concat(dataframes, ignore_index=True)
        return df_flattened
    
    
    def extend_list_all(self, data, max_size):
        new_dat = data.tolist()
        for i in range(len(data)):
            new_dat[i] = self.extend_list(new_dat[i], max_size)
        return np.array(new_dat)

    def extend_dataframe(self, df, col, index):
        df[col] = df[col].apply(lambda x: self.extend_list_all(x, index))
        df[col] = df[col].apply(lambda x: x[:, :index+1])
        #df[col] = df[col][:, :index+1]
        return df
    
    def summarize_metric(self, df, col, auc, index):
        if auc:
            df[col] = df[col].apply(lambda x: (np.trapz(x, axis=1)/x.shape[1]).reshape(-1, 1))
        else:
            df[col] = df[col].apply(lambda x: x[:, index].reshape(-1, 1))
        return df
    
    def summarize_AL_procedure(self, df, auc=True, method="auc_max_ind", indices=[], threshold=1):
        if not auc and method == "auc_custom_ind" and len(indices) == 0:
            print("Need to specify indices for non-auc method with custom indices")
            return None
        df = df.copy()
        for metric in self.metrics:
            col = "mean_" + metric
            df[col] = df[metric].apply(lambda x: np.mean(x, axis=0)) # axis 1 since we transpose in read_all_data (i.e., shape is (num_iterations, num_repeats))
            df['array_lengths'] = df[col].apply(lambda x: len(x))
            min_length = df['array_lengths'].min()
            max_length = df['array_lengths'].max()
            df['last_index_above_threshold'] = df[col].apply(lambda x: np.where(x > threshold)[0][0] if any(x > threshold) else len(x))
            max_index = df['last_index_above_threshold'].max()
            min_index = df['last_index_above_threshold'].min()
            #print("IASDJOASIDJIOASJD")

            if method == "batch_size":
                ind_step = 2000
                df = self.extend_dataframe(df, metric, max_length)
                #print(len(df))
                for i, row in df.iterrows():
                    N = len(self.Y)
                    n_edges = (N*(N-1))/2
                    batch_size = row["num_feedback"]
                    #print(batch_size)
                    if batch_size < 1:
                        continue
                    indices = []
                    for j in range(1, max_index):
                        if (j * batch_size) % ind_step == 0:
                            indices.append(j)
                        if (j*batch_size)/n_edges > 0.6:
                            break
                    indices = np.array(indices)
                    
                    #print("ASD: ", batch_size, indices, len(indices))
                    #print(df.at[i, metric].shape)
                    #print(df.at[i, metric][:, indices])
                    #print(np.mean(df.at[i, metric][:, indices], axis=1).reshape(-1, 1))
                    df.at[i, metric] = np.mean(df.at[i, metric][:, indices], axis=1).reshape(-1, 1)
                #df = self.summarize_metric(df, metric, auc, max_length)
            elif method == "auc_max_ind":
                df = self.extend_dataframe(df, metric, max_length)
                df = self.summarize_metric(df, metric, auc, max_length)
            elif method == "auc_max_thresh":
                df = self.extend_dataframe(df, metric, max_index)
                df = self.summarize_metric(df, metric, auc, max_index)
            elif method == "auc_min_ind":
                df = self.extend_dataframe(df, metric, min_length)
                df = self.summarize_metric(df, metric, auc, min_length)
            elif method == "auc_min_thresh":
                df = self.extend_dataframe(df, metric, min_index)
                df = self.summarize_metric(df, metric, auc, min_index)
            elif method == "auc_custom_ind":
                df = self.extend_dataframe(df, metric, max_length)
                if auc:
                    max_ind = np.max(indices)
                    df = self.extend_dataframe(df, metric, max_ind)
                    df = self.summarize_metric(df, metric, auc, max_ind)
                else:
                    df = self.extend_dataframe(df, metric, max_length)
                    df[metric] = df[metric].apply(lambda x: x[:, indices])
                    df[metric] = df[metric].apply(lambda x: np.mean(x, axis=1).reshape(-1, 1))
            else:
                raise ValueError("Invalid method")
        return df
        
    def get_keys_from_options(self, config):
        options_keys = []
        options_values = []
        for key1, value1 in config.items():
            for key2, value2 in value1.items():
                if type(value2) != list:
                    value2 = [value2]
                options_keys.append(key2)
                options_values.append(value2)
        return options_keys, options_values

    def filter_dataframe(self, df, conditions):
        mask = pd.Series(True, index=df.index)
        for column, values in conditions.items():
            if type(values) != list:
                values = [values]
            mask = mask & df[column].isin(values)
        return df[mask]

    def generate_AL_curves(
        self,
        data,
        save_location,
        categorize,
        compare,
        vary,
        auc, 
        summary_method, 
        indices, 
        threshold, 
        err_style="band",
        marker=None,
        markersize=6,
        capsize=6,
        linestyle="solid",
        **config):
        #data = self.read_all_data(folder="../experiment_results/maxexp_experiment")
        config = copy.deepcopy(config)
        options_keys = []
        options_values = []
        compare_options = {}
        vary_options = {}
        all_options = {}

        for option_category, options in config.items():
            if option_category == "general_options":
                continue
            for option, option_value in options.items():
                if type(option_value) != list:
                    option_value = [option_value]
                all_options[option] = option_value
                if option not in compare:
                    options_keys.append(option)
                    options_values.append(option_value)

        for option in compare:
            compare_options[option] = all_options[option]
            all_options.pop(option, None)

        if "x" not in vary:
            for option in vary:
                vary_options[option] = all_options[option]
                all_options.pop(option, None)
            data = self.summarize_AL_procedure(
                data,
                auc=auc, 
                method=summary_method, 
                indices=indices, 
                threshold=threshold
            )
        
        # extending
        for metric in self.metrics:
            col = "mean_" + metric
            data[col] = data[metric].apply(lambda x: np.mean(x, axis=0)) # axis 1 since we transpose in read_all_data (i.e., shape is (num_iterations, num_repeats))
            data['array_lengths'] = data[col].apply(lambda x: len(x))
            max_length = data['array_lengths'].max()
            data = self.extend_dataframe(data, metric, max_length)

        data_column_names = self.metrics
        non_data_column_names = list(set(data.columns) - set(data_column_names))
        data = self.flatten_dataframe(data, non_data_column_names, data_column_names)

        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))

            for option in compare:
                exp_kwargs[option] = compare_options[option]

            if "x" not in vary:
                for option in vary:
                    exp_kwargs[option] = vary_options[option]

            for metric in self.metrics:
                exp_kwargs["metric"] = metric
                df_filtered = self.filter_dataframe(data, exp_kwargs)

                path = save_location + "/" + metric + "/"
                for option in categorize:
                    path += str(exp_kwargs[option]) + "/" 
                fig_path = Path(path)
                fig_path.mkdir(parents=True, exist_ok=True)
                #file_name = exp_kwargs["dataset"] + "_" + metric + "_" + str(exp_kwargs["noise_level"]) + "_" + exp_kwargs["sim_init_type"] + ".png"
                file_name = exp_kwargs["dataset"] + "_" + metric + "_" + str(exp_kwargs["noise_level"]) + "_" + str(exp_kwargs["sim_init_type"]) + ".png"
                #file_path = path + "plot.png"
                file_path = path + file_name

                # Cont
                #hues = list(all_options.keys())
                #hues.extend(list(compare_options.keys()))
                hues = list(compare_options.keys())
                sns.set_theme()
                sns.set_style("whitegrid")
                SMALL_SIZE = 16
                MEDIUM_SIZE = 18
                BIGGER_SIZE = 18

                plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
                plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
                plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
                plt.rc('figure', dpi=200)
                plt.rc('figure', figsize=(6, 6))

                if err_style == "bars":
                    err_kws = {
                        "capsize": capsize,
                        "marker": marker,
                        "markersize": markersize,
                    }
                else:
                    err_kws={}

                #cut_threshold = 0
                cut_axis = False
                #errorbar = ("sd", 0.3)
                errorbar = ("sd", 1)
                #if "synthetic" in self.dataset:
                #    cut_threshold = 700
                #elif self.dataset == "20newsgroups":
                #    cut_threshold = 700
                #elif self.dataset == "breast_cancer":
                #    # noise = 0.0
                #    errorbar = None
                #    cut_axis = True
                #    l1 = 0
                #    l2 = 0.7
                #    l3 = 0.72
                #    l4 = 1.01
                #    cut_threshold = 700
                #    #cut_threshold = 900
                #elif self.dataset == "cardiotocography":
                #    cut_threshold = 700
                #elif self.dataset == "cifar10" or self.dataset == "cifar10_original":
                #    cut_threshold = 700
                #elif self.dataset == "ecoli":
                #    cut_threshold = 700
                #elif self.dataset == "forest_type_mapping":
                #    cut_threshold = 700
                #elif self.dataset == "mnist":
                #    cut_threshold = 700
                #elif self.dataset == "mushrooms":
                #    cut_threshold = 700
                #elif self.dataset == "user_knowledge":
                #    cut_threshold = 700
                #elif self.dataset == "yeast":
                #    cut_threshold = 700
                #else:
                #    raise ValueError("incorrect dataset!")

                df_filtered = df_filtered[df_filtered[vary[0]] < 1000]
                metric_map = {"ami": "AMI", "rand": "ARI", "time": "Time (s)", "num_violations": "Num. violations"}

                if not cut_axis:
                    ax = sns.lineplot(
                        x=vary[0],
                        y="y",
                        hue=df_filtered[hues].apply(tuple, axis=1),
                        #hue="acq_fn",
                        #hue_order=["maxexp2", "maxmin2", "uncert", "freq", "unif", "nCOBRAS", "COBRAS", "QECC"],
                        errorbar=errorbar,
                        err_style=err_style,
                        data=df_filtered,
                        linestyle=linestyle,
                        err_kws=err_kws,
                    )
                    plt.ylabel(metric_map[metric])
                else:
                    f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False, gridspec_kw={'height_ratios': [3, 1]})
                    ax = sns.lineplot(
                        x=vary[0],
                        y="y",
                        #hue=df_filtered[hues].apply(tuple, axis=1),
                        hue="acq_fn",
                        hue_order=["maxexp", "maxmin", "uncert", "freq", "unif", "nCOBRAS", "COBRAS", "QECC"],
                        errorbar=errorbar,
                        err_style=err_style,
                        data=df_filtered,
                        linestyle=linestyle,
                        err_kws=err_kws,
                        ax=ax1
                    )
                    ax = sns.lineplot(
                        x=vary[0],
                        y="y",
                        #hue=df_filtered[hues].apply(tuple, axis=1),
                        hue="acq_fn",
                        hue_order=["maxexp", "maxmin", "uncert", "freq", "unif", "nCOBRAS", "COBRAS", "QECC"],
                        errorbar=errorbar,
                        err_style=err_style,
                        data=df_filtered,
                        linestyle=linestyle,
                        err_kws=err_kws,
                        ax=ax2
                    )
                    ax1.spines['bottom'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax1.set_ylim(l3, l4)
                    ax2.set_ylim(l1, l2)

                    d = .015  # how big to make the diagonal lines in axes coordinates
                    # arguments to pass to plot, just so we don't keep repeating them
                    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
                    ax1.set_ylabel("")
                    ax2.set_ylabel("")
                    #f.text(metric_map[metric])
                    f.text(-0.02, 0.5, metric_map[metric], va='center', rotation='vertical')


                    #plt.subplots_adjust(wspace=0, hspace=0)
                N = len(self.Y)
                n_edges = (N*(N-1))/2
                if self.num_feedback < 1:
                    batch_size = math.ceil(n_edges * self.num_feedback)
                else:
                    batch_size = self.num_feedback
                n_iterations = int(n_edges/batch_size)
                #tick_labels = np.array(list(range(0, n_iterations))) * batch_size
                #print(batch_size)
                #print(n_edges)
                labels = []
                for item in ax.get_xticks():
                    #print((int(item)*batch_size)/n_edges)
                    labels.append(round((int(item)*batch_size)/n_edges, 2))
                    #labels.append((int(item)*batch_size)/n_edges, 1)



                if not cut_axis:
                    ax.set_xticklabels(labels)
                else:
                    ax2.get_legend().remove()
                    ax2.set_xticklabels(labels)
                    ax = ax1

                #ax.set_xticks(range(n_iterations), labels=tick_labels)
                #plt.xlabel(str(vary))
                plt.xlabel("Proportion of edges queried")
                ax.legend(loc='lower right')

                #legs = ax.get_legend().get_texts()
                ##legs = [l.get_text() for l in legs]
                #fix_legends = False
                #if fix_legends:
                #    #new_legends = []
                #    ax.get_legend().set_title(None)
                #    for ll in legs:
                #        l = ll.get_text()
                #        if "unif" in l:
                #            #new_legends.append("Uniform")
                #            ll.set_text("Uniform")
                #        if "COBRAS" in l and "nCOBRAS" not in l:
                #            #new_legends.append("COBRAS")
                #            ll.set_text("COBRAS")
                #        if "nCOBRAS" in l:
                #            #new_legends.append("nCOBRAS")
                #            ll.set_text("nCOBRAS")
                #        if "freq" in l:
                #            #new_legends.append("Frequency")
                #            ll.set_text("Frequency")
                #        if "uncert" in l:
                #            #new_legends.append("Uncertainty")
                #            ll.set_text("Uncertainty")
                #        if "maxmin" in l:
                #            #new_legends.append("Maxmin")
                #            ll.set_text("Maxmin")
                #        if "maxexp" in l:
                #            #new_legends.append("Maxexp")
                #            ll.set_text("Maxexp")
                #        if "QECC" in l:
                #            #new_legends.append("QECC")
                #            ll.set_text("QECC")
                #    #ax.legend(labels=new_legends)
                legend = ax.get_legend()
                plt.savefig(file_path, bbox_extra_artists=(legend,), bbox_inches='tight')
                plt.savefig(file_path, dpi=200, bbox_inches='tight')
                plt.clf()

    def generate_AL_curves2(
        self,
        data,
        save_location,
        categorize,
        compare,
        vary,
        auc, 
        summary_method, 
        indices, 
        threshold, 
        err_style="band",
        marker=None,
        markersize=6,
        capsize=6,
        linestyle="solid",
        **config):
        #data = self.read_all_data(folder="../experiment_results/maxexp_experiment")
        config = copy.deepcopy(config)
        options_keys = []
        options_values = []
        compare_options = {}
        vary_options = {}
        all_options = {}

        for option_category, options in config.items():
            if option_category == "general_options":
                continue
            for option, option_value in options.items():
                if type(option_value) != list:
                    option_value = [option_value]
                all_options[option] = option_value
                if option not in compare:
                    options_keys.append(option)
                    options_values.append(option_value)

        for option in compare:
            compare_options[option] = all_options[option]
            all_options.pop(option, None)

        if "x" not in vary:
            for option in vary:
                vary_options[option] = all_options[option]
                all_options.pop(option, None)
            data = self.summarize_AL_procedure(
                data,
                auc=auc, 
                method=summary_method, 
                indices=indices, 
                threshold=threshold
            )
        
        # extending
        for metric in self.metrics:
            col = "mean_" + metric
            data[col] = data[metric].apply(lambda x: np.mean(x, axis=0)) # axis 1 since we transpose in read_all_data (i.e., shape is (num_iterations, num_repeats))
            data['array_lengths'] = data[col].apply(lambda x: len(x))
            max_length = data['array_lengths'].max()
            data = self.extend_dataframe(data, metric, max_length)

        data_column_names = self.metrics
        non_data_column_names = list(set(data.columns) - set(data_column_names))
        data = self.flatten_dataframe(data, non_data_column_names, data_column_names)

        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))

            for option in compare:
                exp_kwargs[option] = compare_options[option]

            if "x" not in vary:
                for option in vary:
                    exp_kwargs[option] = vary_options[option]

            for metric in self.metrics:
                exp_kwargs["metric"] = metric
                df_filtered = self.filter_dataframe(data, exp_kwargs)

                path = save_location + "/" + metric + "/"
                for option in categorize:
                    path += str(exp_kwargs[option]) + "/" 
                fig_path = Path(path)
                fig_path.mkdir(parents=True, exist_ok=True)
                file_path = path + "plot.png"

                # Cont
                #hues = list(all_options.keys())
                #hues.extend(list(compare_options.keys()))
                hues = list(compare_options.keys())
                sns.set_theme()
                sns.set_style("whitegrid")
                SMALL_SIZE = 16
                MEDIUM_SIZE = 18
                BIGGER_SIZE = 18

                plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
                plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
                plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
                plt.rc('figure', dpi=200)
                plt.rc('figure', figsize=(6, 6))

                if err_style == "bars":
                    err_kws = {
                        "capsize": capsize,
                        "marker": marker,
                        "markersize": markersize,
                    }
                else:
                    err_kws={}

                errorbar = ("sd", 1)
                ax = sns.lineplot(
                    x=vary[0],
                    y="y",
                    #hue=df_filtered[hues].apply(tuple, axis=1),
                    hue="acq_fn",
                    hue_order=["maxexp2", "maxmin2", "uncert", "freq", "unif", "nCOBRAS", "COBRAS", "QECC"],
                    errorbar=errorbar,
                    err_style=err_style,
                    data=df_filtered,
                    linestyle=linestyle,
                    err_kws=err_kws,
                )
                metric_map = {"ami": "AMI (AUC)", "rand": "ARI (AUC)"}
                plt.ylabel(metric_map[metric])
              
                #ax.set_xticks(range(n_iterations), labels=tick_labels)
                x_label_ = "Noise level" if vary[0] == "noise_level" else "Batch size"
                plt.xlabel(x_label_)
                #plt.xlabel("Proportion of edges queried")
                ax.legend(loc='best')

                legs = ax.get_legend().get_texts()
                #legs = [l.get_text() for l in legs]
                new_legends = []
                ax.get_legend().set_title(None)
                for ll in legs:
                    l = ll.get_text()
                    if "unif" in l:
                        #new_legends.append("Uniform")
                        ll.set_text("Uniform")
                    if "COBRAS" in l and "nCOBRAS" not in l:
                        #new_legends.append("COBRAS")
                        ll.set_text("COBRAS")
                    if "nCOBRAS" in l:
                        #new_legends.append("nCOBRAS")
                        ll.set_text("nCOBRAS")
                    if "freq" in l:
                        #new_legends.append("Frequency")
                        ll.set_text("Frequency")
                    if "uncert" in l:
                        #new_legends.append("Uncertainty")
                        ll.set_text("Uncertainty")
                    if "maxmin" in l:
                        #new_legends.append("Maxmin")
                        ll.set_text("Maxmin")
                    if "maxexp" in l:
                        #new_legends.append("Maxexp")
                        ll.set_text("Maxexp")
                    if "QECC" in l:
                        #new_legends.append("QECC")
                        ll.set_text("QECC-heur")
                #ax.legend(labels=new_legends)
                legend = ax.get_legend()
                plt.savefig(file_path, bbox_extra_artists=(legend,), bbox_inches='tight')
                plt.savefig(file_path, dpi=200, bbox_inches='tight')
                plt.clf()

    
    def generate_experiments(self, folder, options_to_keep, start_index=1, **config):
        options_to_compare_extracted = {}
        all_options = {}
        i = start_index
        fig_path = Path(folder)
        fig_path.mkdir(parents=True, exist_ok=True)
        for option_category, options in config.items():
            if option_category not in all_options:
                all_options[option_category] = list(options.keys())

            if option_category not in options_to_compare_extracted:
                options_to_compare_extracted[option_category] = {}

            for option in options_to_keep:
                if option in config[option_category]:
                    options_to_compare_extracted[option_category][option] = config[option_category][option]
                    config[option_category].pop(option, None)

        options_keys, options_values = self.get_keys_from_options(config)
        for exp_vals in itertools.product(*options_values):
            exp_kwargs = dict(zip(options_keys, exp_vals))
            kwargs = {}
            for option_category, options in all_options.items():
                if option_category not in kwargs:
                    kwargs[option_category] = {}
                for option in options:
                    if option in options_to_keep:
                        continue
                    kwargs[option_category][option] = exp_kwargs[option]

            for option_category, options in options_to_compare_extracted.items():
                if option_category not in kwargs:
                    kwargs[option_category] = {}
                for option, option_values in options.items():
                    kwargs[option_category][option] = option_values

            with open(folder + "/experiment" + str(i) + ".json", "w") as fp:
                json.dump(kwargs, fp, indent=4)
            i += 1
        return i