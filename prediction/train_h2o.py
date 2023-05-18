'''
This reads the input from the .csv file for Axiline SVM.
Trains DL model to predict backend and frontend metrics
for unseen configurations.

For training the model we will be using H2o. This will
utilize the grid search method. Here we will first find
the max depth and then tune the rest of the parameters.
'''

import os
import h2o
import pandas as pd
import numpy as np
import random
from datetime import datetime
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import argparse

config = ['size', 'num_cycle']
feature_list = ['size', 'num_cycle', 'num_unit', 'bit_width', \
                'benchmark_no', 'target_clock_frequency(GHz)', 'util']

## Used functions ##
def gen_test_dfs(test_df, config = config):
    test_configs_df = test_df[config].drop_duplicates().reset_index(drop=True)
    test_dfs = []
    for i in range(len(test_configs_df)):
        print(i,test_configs_df[i:i+1])
        config_df = test_configs_df[i:i+1]
        tmp_df = test_df.merge(config_df, how='inner', on=config)
        test_dfs.append(tmp_df)
    
    return test_dfs

def add_fold_number(train_df, no_fold, seed = 42, config = config):
    train_configs_df = train_df[config].drop_duplicates().reset_index(drop=True)
    no_train_config = len(train_configs_df)
    no_in_each_fold = int(no_train_config/no_fold)
    no_act_fold = int(np.floor(no_train_config/no_in_each_fold))
    fold_clm = []
    
    for i in range(no_act_fold):
        for _ in range(no_in_each_fold):
            fold_clm.append(i)
        remaining_no = no_train_config - len(fold_clm)
        remaining_fold = no_act_fold - i - 1
        if remaining_fold == 0:
            if remaining_no != 0:
                for _ in range(remaining_no):
                    fold_clm.append(i)
        else:
            remaining_each_no = remaining_no / remaining_fold
        
        if remaining_each_no > no_in_each_fold:
            fold_clm.append(i)
    
    random.seed(seed)
    random.shuffle(fold_clm)
    print(len(fold_clm), len(train_configs_df))
    train_configs_df['fold_number'] = fold_clm
    return train_configs_df

def mlReport(y1, y2, isPrint = 1, detail = 0):
    df = pd.DataFrame()
    if type(y1).__name__ == "Series":
        df['act'] = y1.to_numpy()
    elif type(y1).__name__ == "ndarray":
        df['act'] = y1.reshape(-1)
    else:
        df['act'] = np.array(y1).reshape(-1)

    if type(y2).__name__ == "Series":
        df['predict'] = y2.to_numpy()
    elif type(y2).__name__ == "ndarray":
        df['predict'] = y2.reshape(-1)
    else:
        df['predict'] = np.array(y2).reshape(-1)

    df['p_error'] = 100*np.abs(df['predict'] - df['act'])/df['act']
    error = np.mean(df['p_error'].to_numpy())
    std_error = np.std(df['p_error'].to_numpy())
    max_error = np.max(df['p_error'].to_numpy())

    if isPrint == 1:
        print(f'Percentage Error: {error}')
        print(f'STD Percentage Error: {std_error}')
        print(f'Max Percentage Error: {max_error}')

    if detail == 1:
        return error, std_error, df
    elif detail == 2:
        return error, max_error, None
    elif detail == 3:
        return error, max_error, std_error
    return error, std_error, None

def print_error(tests, test_dfs, best_model, metric):
    uAPE = []
    MAPE = []
    for i in range(len(tests)):
        test = tests[i]
        prediction = best_model.predict(test)
        test_df = test_dfs[i]
        test_df[f'predicted_{metric}'] = prediction.as_data_frame()
        error, max_error, std_error = mlReport(test_df[metric],
                                    test_df[f'predicted_{metric}'], 
                                    isPrint = 0, detail = 3)
        print(f'ID:{i} uAPE:{error} MAPE:{max_error} STD:{std_error}')
        uAPE.append(error)
        MAPE.append(max_error)
    
    print(f'Average uAPE: {np.mean(uAPE)}, Std uAPE: {np.std(uAPE)}')
    print(f'Average MAPE: {np.mean(MAPE)}, Std MAPE: {np.std(MAPE)}')
    return

def save_model(model, dir_name, file_name):
    model_path = h2o.save_model(model = model, path = dir_name, force = True)
    os.rename(model_path, f'{dir_name}/{file_name}')
    return

def train_dl_model(train, feature_list = feature_list, 
                   metric = 'energy(uJ)', id = 0):
    
    seed = 42
    stopping_tolerance = 1e-5
    stopping_rounds = 20
    stopping_metric = 'RMSE'
    thread = 16
    epochs = 10000

    ## First generate possible architectures
    dl_layers = []
    for d in range(3,7,1):
        for nc in range(5, 101, 5):
            ll = [nc for i in range(d)]
            dl_layers.append(ll)

    hyper_parameters = {'activation': ['Rectifier', 'Maxout'],
                        'hidden': dl_layers,
                        'rho' : [0.99, 0.9, 0.95, 0.999]}
    
    search_criteria = {
        'strategy': "RandomDiscrete",
        'seed': seed,
        'max_runtime_secs': 600,
        }

    dl_model = H2ODeepLearningEstimator(seed = seed, epochs = epochs,
                                    stopping_tolerance = stopping_tolerance,
                                    stopping_metric = stopping_metric,
                                    stopping_rounds = stopping_rounds,
                                    fold_column = "fold_number",
                                    reproducible = True,
                                    keep_cross_validation_predictions = True,
                                    model_id = f'dl_{id}')

    grid_id = f'dl_grid_{id}'

    ## Grid Search for Max depth
    print('Starting Training')
    dl_grid = H2OGridSearch(model = dl_model, hyper_params = hyper_parameters,
                            grid_id = grid_id, search_criteria = search_criteria)

    dl_grid.train(x = feature_list, y = metric, training_frame = train,
                    seed = seed, parallelism = thread)
    
    print('Training Finished')
    sorted_dl_grid = dl_grid.get_grid(sort_by = stopping_metric, decreasing = False)
    best_model = sorted_dl_grid[0]
    return best_model

def train_rf_model(train, feature_list = feature_list, metric = 'energy(uJ)', id = 0):
    seed = 42
    stopping_rounds = 10
    stopping_tolerance = 1e-4
    stopping_metric = 'RMSE'
    thread = 16
    ntrees = 500
    depth_range = 3

    ## First find the suitable depth value
    hyper_parameters = {'max_depth':[i for i in range(depth_range + 1, 150, depth_range)],
                        'mtries':[i for i in range(1,4)]}

    rf_model = H2ORandomForestEstimator(seed = seed,
                                        stopping_tolerance = stopping_tolerance,
                                        stopping_metric = stopping_metric,
                                        stopping_rounds = stopping_rounds,
                                        fold_column = "fold_number",
                                        keep_cross_validation_predictions = True,
                                        model_id = f'rf_{pid}_{id}')

    grid_id = f'depth_grid_{pid}_{id}'
    search_criteria = {'strategy': "Cartesian"}

    ## Grid Search for Max depth
    rf_grid = H2OGridSearch(model = rf_model, hyper_params = hyper_parameters,
                            grid_id = grid_id, search_criteria = search_criteria)

    rf_grid.train(x = feature_list, y = metric, training_frame = train,
                    seed = seed, parallelism = thread)
    print('Training Finished')

    sorted_rf_depth = rf_grid.get_grid(sort_by = stopping_metric, decreasing = False)
    print(sorted_rf_depth.sorted_metric_table())

    tmp_best_model = sorted_rf_depth[0]

    ## Print Performance on test data ##
    print_error(tests, test_dfs, tmp_best_model, metric)
    print_error([train], [train_df], tmp_best_model, metric)

    mtries = tmp_best_model.actual_params['mtries']
    best_max_depth = tmp_best_model.actual_params['max_depth']
    min_max_depth = max(1, best_max_depth - depth_range)
    max_max_depth = best_max_depth + depth_range + 1

    hyper_parameters = {'max_depth': [i for i in range(min_max_depth, max_max_depth)],
                        'sample_rate': [x/100.0 for x in range(20, 101)]
                        }

    rf_main_model = H2ORandomForestEstimator(seed = seed, mtries = mtries,
                                        ntrees = ntrees,
                                        stopping_tolerance = stopping_tolerance,
                                        stopping_metric = stopping_metric,
                                        stopping_rounds = stopping_rounds,
                                        fold_column = "fold_number",
                                        keep_cross_validation_predictions = True,
                                        model_id = f'rf_main_{id}')

    grid_main_id = f'rf_random_grid_{id}'
    search_criteria = {"strategy":"RandomDiscrete",
                    "max_runtime_secs":300,
                    "seed":seed}

    rf_main_grid = H2OGridSearch(model = rf_main_model, 
                            hyper_params = hyper_parameters, 
                            grid_id = grid_main_id,
                            search_criteria = search_criteria)

    rf_main_grid.train(x = feature_list, y = metric, training_frame = train,
                    seed = seed, parallelism = thread)
    print('Training Finished')

    sorted_rf_model = rf_main_grid.get_grid(sort_by = stopping_metric, decreasing = False)

    #print(sorted_rf_model.sorted_metric_table())

    best_model = sorted_rf_model[0]
    return best_model

def train_xgb_model(train, feature_list = feature_list, metric = 'energy(uJ)', id = 0):
    seed = 42
    stopping_rounds = 10
    stopping_tolerance = 1e-4
    stopping_metric = 'RMSE'
    thread = 16
    ntrees = 300
    depth_range = 3

    ## First find the suitable depth value
    hyper_parameters = {'max_depth':[i for i in range(5, 100, depth_range)]}

    gbm_model = H2OGradientBoostingEstimator(seed = seed, ntrees = ntrees,
                                    stopping_tolerance = stopping_tolerance,
                                    stopping_metric = stopping_metric,
                                    stopping_rounds = stopping_rounds,
                                    fold_column = "fold_number",
                                    keep_cross_validation_predictions = True,
                                    model_id = f'gbm_{id}')

    grid_id = f'depth_grid_{id}'
    search_criteria = {'strategy': "Cartesian"}

    ## Grid Search for Max depth
    gbm_grid = H2OGridSearch(model = gbm_model, hyper_params = hyper_parameters,
                            grid_id = grid_id, search_criteria = search_criteria)

    gbm_grid.train(x = feature_list, y = metric, training_frame = train,
                    seed = seed, parallelism = thread)
    print('Training Finished')

    sorted_gbm_depth = gbm_grid.get_grid(sort_by = stopping_metric, decreasing = False)
    print(sorted_gbm_depth.sorted_metric_table())

    tmp_best_model = sorted_gbm_depth[0]
    ## Print Performance on test data ##
    print_error(tests, test_dfs, tmp_best_model, metric)
    print_error([train], [train_df], tmp_best_model, metric)

    best_max_depth = tmp_best_model.actual_params['max_depth']
    min_max_depth = max(1, best_max_depth - depth_range)
    max_max_depth = best_max_depth + depth_range + 1

    hyper_parameters = {'max_depth': [i for i in range(min_max_depth, max_max_depth)],
                        'sample_rate': [x/100.0 for x in range(20, 101)]}

    gbm_main_model = H2OGradientBoostingEstimator(seed = seed,
                                        ntrees = ntrees,
                                        stopping_tolerance = stopping_tolerance,
                                        stopping_metric = stopping_metric,
                                        stopping_rounds = stopping_rounds,
                                        fold_column = "fold_number",
                                        keep_cross_validation_predictions = True,
                                        model_id = f'gbm_main_{pid}_{id}')

    grid_main_id = f'gbm_random_grid_{id}'
    search_criteria = {"strategy":"RandomDiscrete",
                    "max_runtime_secs":300,
                    "seed":seed}

    gbm_main_grid = H2OGridSearch(model = gbm_main_model,
                            hyper_params = hyper_parameters,
                            grid_id = grid_main_id,
                            search_criteria = search_criteria)

    gbm_main_grid.train(x = feature_list, y = metric, training_frame = train,
                    seed = seed, parallelism = thread)
    print('Training Finished')

    sorted_gbm_model = gbm_main_grid.get_grid(sort_by = stopping_metric, decreasing = False)

    print(sorted_gbm_model.sorted_metric_table())

    best_model = sorted_gbm_model[0]
    return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
                        "Provide details of the required input")
    parser.add_argument('--sample', '-s', default= "LHS", 
                    required=False,
                    help="Enter which sampling method: LHS, halton, sobol")
    parser.add_argument('--count', '-c',
                    default=24,
                    required=False,
                    help="Enter the number of samples: 16, 24, 32")
    parser.add_argument('--metric', '-m', 
                    default='energy(uJ)',
                    required=False,
                    help="Provide metric for which to train")
    parser.add_argument('--MLAlgo', '-M',
                    default='DL',
                    help="Provide which ML model to train. Options: DL, RF, XGB")

    parser.add_argument('--foldCount', '-f', default=4, type=int,
                    help="Provide number of fold for cross validation")

    parser.add_argument('--output', '-o',
                    default='./Output',
                    help='Provide the output directory')
    parser.add_argument('--uniqeFold', '-u', action='store_true',
                    help="If cross validation folds are uniqe")
    args = parser.parse_args()
    
    #['effective_clock_frequency(GHz)', 'total_power(mW)', 'runtime(ms)', 'energy(uJ)']
    data_dir = '/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/test/data/'
    test_file = f'{data_dir}/axiline_svm_sampling_exp_test.csv'
    trian_file = f'{data_dir}/axiline_svm_sampling_exp_train_{args.sample}_{args.count}.csv'

    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(trian_file)
    metric = args.metric

    train_config_df = add_fold_number(train_df, args.foldCount)
    train_df = train_df.merge(train_config_df, how='inner', on=config)
    test_dfs = gen_test_dfs(test_df)

    ## Initialisze H2O server ##
    pid = os.getpid()
    h2o_server_name = f'RTML_{pid}'
    h2o.init(nthreads = 16, max_mem_size = "20G", name = h2o_server_name)
    train=h2o.H2OFrame(train_df)
    tests = []

    for test_df in test_dfs:
        test = h2o.H2OFrame(test_df)
        tests.append(test)

    print(f'Metric is {metric}')
    if args.MLAlgo == 'DL':
        best_model = train_dl_model(train, feature_list, metric = metric)
    elif args.MLAlgo == 'RF':
        best_model = train_rf_model(train, feature_list, metric)
    elif args.MLAlgo == 'XGB':
        best_model = train_xgb_model(train, feature_list, metric)
    else:
        print('Provide correct Alog')
        exit()
    
    ## Print Performance on test data ##
    print_error(tests, test_dfs, best_model, metric)
    print_error([train], [train_df], best_model, metric)
    save_model(best_model, args.output, \
        f'{args.MLAlgo}_{args.sample}_{args.count}_{args.foldCount}_{metric}')