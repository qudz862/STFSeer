import os
import sys
sys.path.append('./')
# from ray import tune
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.tune.suggest.bayesopt import BayesOptSearch
# from ray.tune.suggest.basic_variant import BasicVariantGenerator
# from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
# from ray.tune.suggest import ConcurrencyLimiter
import json
import copy
import torch
import random
import math
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from analysis.event_identify import compute_meta_attrs
import importlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from libcity.data.point_metadata import PointMetaData
from libcity.data.phases import Phases
from libcity.data.entities_events import Entities_Events
from libcity.data.anomalies import Anomalies
from libcity.evaluator.slice_evaluator import SliceEvaluator
from libcity.evaluator.residuals_distribution import ResidualsDistribution
from libcity.evaluator.phases_evaluator import PhaseEvaluator
from libcity.evaluator.entities_events_evaluator import EntitiesEventsEvaluator


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    # print('config ~ ', config.config)
    config.config['add_time_in_day'] = False
    exp_id = config.get('exp_id', None)
    train = config.config.get('train', None)
    
    print('train', train)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    # 将config写到文件，存入模型的cache文件夹中
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    model_config = copy.deepcopy(config.config)
    del model_config['device']
    ensure_dir('./libcity/cache/{}'.format(exp_id))
    model_config_file = './libcity/cache/{}/model_config.json'.format(exp_id)
    # 使用 json.dump 将字典保存为 JSON 文件
    with open(model_config_file, 'w') as config_json_file:
        json.dump(model_config, config_json_file)
    
    print('config: ', config, config.config)
    
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    
    # 转换数据，并划分数据集
    data_aug_flag = model_config.get('augmented_data', False)
    eval_flag = model_config.get('eval', True)
    if data_aug_flag:
        original_train_data, train_data, valid_data, test_data = dataset.get_data()
    else:
        train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    
    # 加载执行器
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    # 训练
    if train or not os.path.exists(model_cache_file):
        print('train', train)
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    train_truths, train_preds, valid_truths, valid_preds, test_truths, test_preds = None, None, None, None, None, None
    if eval_flag:
        if data_aug_flag:
            print("evaluating aug model with original train data")
            train_truths, train_preds = executor.evaluate(original_train_data, 'train')
        else:
            train_truths, train_preds = executor.evaluate(train_data, 'train')
        valid_truths, valid_preds = executor.evaluate(valid_data, 'valid')
        test_truths, test_preds = executor.evaluate(test_data, 'test')
    else:
        pred_truth_path = './libcity/cache/' + str(exp_id) + '/evaluate_cache/'
        ensure_dir(pred_truth_path)
        pred_truth_file_name_train = pred_truth_path + model_name + '_' + dataset_name + '_train_predictions.npz'
        pred_truth_file_name_valid = pred_truth_path + model_name + '_' + dataset_name + '_valid_predictions.npz'
        pred_truth_file_name_test = pred_truth_path + model_name + '_' + dataset_name + '_test_predictions.npz'
        train_results = np.load(pred_truth_file_name_train)
        valid_results = np.load(pred_truth_file_name_valid)
        test_results = np.load(pred_truth_file_name_test)
        train_truths = train_results['truth']
        train_preds = train_results['prediction']
        valid_truths = valid_results['truth']
        valid_preds = valid_results['prediction']
        test_truths = test_results['truth']
        test_preds = test_results['prediction']
    all_truths = np.concatenate((train_truths, valid_truths, test_truths), axis=0)
    all_preds = np.concatenate((train_preds, valid_preds, test_preds), axis=0)
    # all_residuals = all_truths - all_preds
    all_residuals = all_preds - all_truths
    print('all_residuals shape: ', all_residuals.shape)
    all_truths_series = instances_to_series(all_truths[...,0])
    all_preds_series = instances_to_series(all_preds[...,0])
    all_residuals_series = instances_to_series(all_residuals[...,0])
    
    # generate forecast scopes or sampled forecast steps
    forecast_scope_config = config['forecast_scope']
    forecast_scopes = []
    if forecast_scope_config['scope_mode'] == 'range':
        for start in range(0, model_config['output_window'], forecast_scope_config['step_unit']):
            end = start + forecast_scope_config['step_unit']
            if end > model_config['output_window']: end = model_config['output_window']
            forecast_scopes.append([start, end])
    elif forecast_scope_config['scope_mode'] == 'sample':
        if forecast_scope_config['sample_method'] == 'linear':
            samples = np.linspace(0, model_config['output_window']-1, forecast_scope_config['sample_num']).tolist()
            forecast_scopes = [[round(sample), round(sample)+1] for sample in samples]
        elif forecast_scope_config['sample_method'] == 'exponential':
            exp_intervals = np.linspace(0, 1, forecast_scope_config['sample_num']) ** 2
            samples = exp_intervals * (model_config['output_window']-1)
            samples = samples.tolist()
            forecast_scopes = [[round(sample), round(sample)+1] for sample in samples]
    # compute distribution of residuals
    residual_dis_cls = ResidualsDistribution(model_config, all_residuals, forecast_scopes)
    residual_bins = residual_dis_cls.compute_residual_bins()
    residual_dis_cls.compute_residual_boxes()
    residual_hists = residual_dis_cls.compute_residual_distribution()
    residual_dis_cls.save_residual_distributions()
    
    # 执行完全部的过程后，将数据和模型的信息添加到existed_task_data_model.json中
    # 首先读入当前的existed_task_data_model.json
    with open('./existed_task_data_model.json', 'r') as f:
        existed_task_data_model = json.load(f)
    window_str = '{}-{}-{}'.format(model_config['input_window'], model_config['output_window'], model_config['output_offset'])
    if task not in existed_task_data_model:
        existed_task_data_model[task] = {}
    if dataset_name not in existed_task_data_model[task]:
        existed_task_data_model[task][dataset_name] = {}
        with open('./dataset_infor.json', 'r') as f:
            dataset_infor = json.load(f)
        existed_task_data_model[task][dataset_name]['data_type'] = dataset_infor[dataset_name]['Data_type']
    if window_str not in existed_task_data_model[task][dataset_name]['forecast_steps']:
        existed_task_data_model[task][dataset_name]['forecast_steps'][window_str] = []
    specific_model_name = model_name + '-' + str(exp_id)
    if specific_model_name not in existed_task_data_model[task][dataset_name]['forecast_steps'][window_str]:
        existed_task_data_model[task][dataset_name]['forecast_steps'][window_str].append(specific_model_name)
    # 将existed_task_data_model写入json文件
    with open('./existed_task_data_model.json', 'w') as f:
        json.dump(existed_task_data_model, f)
    print('writing existed_task_data_model.json')
    
    dataset_path = './libcity/cache/dataset_cache/'
    raw_dataset_file_name = dataset_path + 'raw_air_quality_' + dataset_name + '.npy'
    raw_dataset = np.load(raw_dataset_file_name)
    # extract focus phases
    time_strs = dataset.timesolts
    geo_coords = dataset.geo_coords
    print(f"focus value threshold: {config['focus_th']}")
    print("Computing focus phases...")
    phase_params = config.get('phase_params', None)
    phases_cls = Phases(config, raw_dataset, specific_model_name, time_strs, geo_coords, phase_params)
    phases_cls.get_all_phases()
    # compute entities and events
    print("Computing focus entities and events...")
    event_params = config.get('event_params', None)
    entities_events_cls = Entities_Events(config, raw_dataset, phases_cls.all_phases, geo_coords, event_params)
    phases_cls.all_phases = entities_events_cls.get_all_entities_events()
    entities_events_cls.save_entities_events()
    print("Entities and events have been saved.")
    
    entities_events_eval_cls = EntitiesEventsEvaluator(config, forecast_scopes, raw_dataset, all_truths_series, all_preds_series, phases_cls.all_phases, entities_events_cls.step_entities, entities_events_cls.all_events, geo_coords)
    entities_events_eval_cls.compute_phase_entities_errors()
    
    
    phase_eval_cls = PhaseEvaluator(raw_dataset, specific_model_name, config, phases_cls.all_phases, all_truths_series, all_preds_series, all_residuals_series, residual_bins, forecast_scopes)
    phase_eval_cls.compute_phases_err_metrics()
    phase_eval_cls.compute_phases_attr_metric_bins()
    phases_cls.save_phases()
    print("Phases have been saved.")
    # compute point metadata for different data structures, and save them as files
    print("Computing point metadata...")
    point_metadata = None
    if config['point_metadata'] != []:
        point_metadata = PointMetaData(model_config, raw_dataset)
        point_metadata.compute_point_metadata()
        print("Point metadata have been saved.")
    print("Computing data slices...")
    if isinstance(config['slice_params']['data_scope'], list):
        for scope in config['slice_params']['data_scope']:
            slices_cls = SliceEvaluator(config, scope, point_metadata.meta_data_series, all_residuals_series, residual_bins, phases_cls.all_phases, forecast_scopes)
            slices_cls.subgroup_mining()
            slices_cls.save_ranges_slices()
    else:
        slices_cls = SliceEvaluator(config, config['slice_params']['data_scope'], point_metadata.meta_data_series, all_residuals_series, residual_bins, phases_cls.all_phases, forecast_scopes)
        slices_cls.subgroup_mining()
        slices_cls.save_ranges_slices()
    print("Slices have been saved.")
    print("Computing anomalies...")
    anomaly_cls = Anomalies(config, raw_dataset)
    anomaly_cls.extract_anomalies()
    print('finish~')

# 元数据可以分为多种的：针对时空位置、长期情况
# 时空事件的元数据？？这个需要根据阈值来设置，跟前端交互来设置。

    
# 将阶段与事件的提取，放到后端来进行
def compute_entity_metadata(meta_attrs, config, data_name, input_window, output_window):
    focus_th = config['focus_th']
    metadata_dir = f'./meta_data/{data_name}/'
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    metadata_raw_file = f'{metadata_dir}/entity_metadata_raw_{input_window}_{output_window}_{focus_th}.npz'
    if os.path.exists(metadata_raw_file):
        return 'ok'
    

def instances_to_series(data_instances):
    ins_num, output_window, loc_num = data_instances.shape
    error_step_num = ins_num + output_window-1
    data_series = []
    for i in range(error_step_num):
        data_series.append([])
        for j in range(loc_num):
            data_series[i].append([])
    data_series_size = 0
    for i in range(ins_num):
        for j in range(output_window):
            for k in range(loc_num):
                data_series[i+j][k].append(data_instances[i][j][k])
                data_series_size += 1
    
    for i in range(len(data_series)):
        for j in range(len(data_series[0])):
            data_series[i][j].reverse()
    print('data_series_size: ', data_series_size)
    
    return data_series

# def instances_to_series(data_instances):
#     ins_num, output_window, loc_num = data_instances.shape
#     error_step_num = ins_num + output_window-1
#     data_series = np.full((error_step_num, loc_num, output_window), np.nan)
#     for i in range(ins_num):
#         for j in range(output_window):
#             seq_len = output_window
#             if i < output_window: seq_len = j
#             if i > error_step_num - output_window: seq_len = output_window - j
#             data_series[i:i+seq_len, :, j] = data_instances[i, j, :]
#     print('data_series array', data_series.size)
#     data_series = data_series[:, :, ::-1].tolist()
#     data_series_size = 0
#     for i in range(len(data_series)):
#         if i > output_window and i < error_step_num - output_window: continue
#         for j in range(len(data_series[i])):
#             filtered_lst = [x for x in data_series[i][j] if not np.isnan(x)]
#             data_series[i][j] = filtered_lst
#             data_series_size += output_window - len(filtered_lst)
#     print('data_series list', data_series_size)
#     return data_series

# def parse_search_space(space_file):
#     search_space = {}
#     if os.path.exists('./{}.json'.format(space_file)):
#         with open('./{}.json'.format(space_file), 'r') as f:
#             paras_dict = json.load(f)
#             for name in paras_dict:
#                 paras_type = paras_dict[name]['type']
#                 if paras_type == 'uniform':
#                     # name type low up
#                     try:
#                         search_space[name] = tune.uniform(paras_dict[name]['lower'], paras_dict[name]['upper'])
#                     except:
#                         raise TypeError('The space file does not meet the format requirements,\
#                             when parsing uniform type.')
#                 elif paras_type == 'randn':
#                     # name type mean sd
#                     try:
#                         search_space[name] = tune.randn(paras_dict[name]['mean'], paras_dict[name]['sd'])
#                     except:
#                         raise TypeError('The space file does not meet the format requirements,\
#                             when parsing randn type.')
#                 elif paras_type == 'randint':
#                     # name type lower upper
#                     try:
#                         if 'lower' not in paras_dict[name]:
#                             search_space[name] = tune.randint(paras_dict[name]['upper'])
#                         else:
#                             search_space[name] = tune.randint(paras_dict[name]['lower'], paras_dict[name]['upper'])
#                     except:
#                         raise TypeError('The space file does not meet the format requirements,\
#                             when parsing randint type.')
#                 elif paras_type == 'choice':
#                     # name type list
#                     try:
#                         search_space[name] = tune.choice(paras_dict[name]['list'])
#                     except:
#                         raise TypeError('The space file does not meet the format requirements,\
#                             when parsing choice type.')
#                 elif paras_type == 'grid_search':
#                     # name type list
#                     try:
#                         search_space[name] = tune.grid_search(paras_dict[name]['list'])
#                     except:
#                         raise TypeError('The space file does not meet the format requirements,\
#                             when parsing grid_search type.')
#                 else:
#                     raise TypeError('The space file does not meet the format requirements,\
#                             when parsing an undefined type.')
#     else:
#         raise FileNotFoundError('The space file {}.json is not found. Please ensure \
#             the config file is in the root dir and is a txt.'.format(space_file))
#     return search_space


# def hyper_parameter(task=None, model_name=None, dataset_name=None, config_file=None, space_file=None,
#                     scheduler=None, search_alg=None, other_args=None, num_samples=5, max_concurrent=1,
#                     cpu_per_trial=1, gpu_per_trial=1):
#     """ Use Ray tune to hyper parameter tune

#     Args:
#         task(str): task name
#         model_name(str): model name
#         dataset_name(str): dataset name
#         config_file(str): config filename used to modify the pipeline's
#             settings. the config file should be json.
#         space_file(str): the file which specifies the parameter search space
#         scheduler(str): the trial sheduler which will be used in ray.tune.run
#         search_alg(str): the search algorithm
#         other_args(dict): the rest parameter args, which will be pass to the Config
#     """
#     # load config
#     experiment_config = ConfigParser(task, model_name, dataset_name, config_file=config_file,
#                                      other_args=other_args)
#     # exp_id
#     exp_id = experiment_config.get('exp_id', None)
#     if exp_id is None:
#         exp_id = int(random.SystemRandom().random() * 100000)
#         experiment_config['exp_id'] = exp_id
#     # logger
#     logger = get_logger(experiment_config)
#     logger.info('Begin ray-tune, task={}, model_name={}, dataset_name={}, exp_id={}'.
#                 format(str(task), str(model_name), str(dataset_name), str(exp_id)))
#     logger.info(experiment_config.config)
#     # check space_file
#     if space_file is None:
#         logger.error('the space_file should not be None when hyperparameter tune.')
#         exit(0)
#     # seed
#     seed = experiment_config.get('seed', 0)
#     set_random_seed(seed)
#     # parse space_file
#     search_sapce = parse_search_space(space_file)
#     # load dataset
#     dataset = get_dataset(experiment_config)
#     # get train valid test data
#     train_data, valid_data, test_data = dataset.get_data()
#     data_feature = dataset.get_data_feature()

#     def train(config, checkpoint_dir=None, experiment_config=None,
#               train_data=None, valid_data=None, data_feature=None):
#         """trainable function which meets ray tune API

#         Args:
#             config (dict): A dict of hyperparameter.
#         """
#         # modify experiment_config
#         for key in config:
#             if key in experiment_config:
#                 experiment_config[key] = config[key]
#         experiment_config['hyper_tune'] = True
#         logger = get_logger(experiment_config)
#         # exp_id
#         exp_id = int(random.SystemRandom().random() * 100000)
#         experiment_config['exp_id'] = exp_id
#         logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
#                     format(str(task), str(model_name), str(dataset_name), str(exp_id)))
#         logger.info('running parameters: ' + str(config))
#         # load model
#         model = get_model(experiment_config, data_feature)
#         # load executor
#         executor = get_executor(experiment_config, model, data_feature)
#         # checkpoint by ray tune
#         if checkpoint_dir:
#             checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
#             executor.load_model(checkpoint)
#         # train
#         executor.train(train_data, valid_data)

#     # init search algorithm and scheduler
#     if search_alg == 'BasicSearch':
#         algorithm = BasicVariantGenerator()
#     elif search_alg == 'BayesOptSearch':
#         algorithm = BayesOptSearch(metric='loss', mode='min')
#         # add concurrency limit
#         algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
#     elif search_alg == 'HyperOpt':
#         algorithm = HyperOptSearch(metric='loss', mode='min')
#         # add concurrency limit
#         algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
#     else:
#         raise ValueError('the search_alg is illegal.')
#     if scheduler == 'FIFO':
#         tune_scheduler = FIFOScheduler()
#     elif scheduler == 'ASHA':
#         tune_scheduler = ASHAScheduler()
#     elif scheduler == 'MedianStoppingRule':
#         tune_scheduler = MedianStoppingRule()
#     else:
#         raise ValueError('the scheduler is illegal')
#     # ray tune run
#     ensure_dir('./libcity/cache/hyper_tune')
#     result = tune.run(tune.with_parameters(train, experiment_config=experiment_config, train_data=train_data,
#                       valid_data=valid_data, data_feature=data_feature),
#                       resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial}, config=search_sapce,
#                       metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
#                       local_dir='./libcity/cache/hyper_tune', num_samples=num_samples)
#     best_trial = result.get_best_trial("loss", "min", "last")
#     logger.info("Best trial config: {}".format(best_trial.config))
#     logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
#     # save best
#     best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
#     model_state, optimizer_state = torch.load(best_path)
#     model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
#         exp_id, model_name, dataset_name)
#     ensure_dir('./libcity/cache/{}/model_cache'.format(exp_id))
#     torch.save((model_state, optimizer_state), model_cache_file)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
