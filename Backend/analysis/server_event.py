import sys
sys.path.append('./')
from libcity.utils import get_executor, get_model, ensure_dir, set_random_seed
from libcity.data import get_dataset
from collections import OrderedDict

from flask import Flask,json,jsonify
from flask_cors import *
import pandas as pd
import pymongo
import datetime
import numpy as np
import math
import random
import os
import json
import copy
import hashlib

from libcity.data.phases import Phases
from libcity.data.entities_events import Entities_Events
from libcity.data.point_metadata import PointMetaData
from libcity.evaluator.slice_evaluator import SliceEvaluator
from libcity.evaluator.phases_evaluator import PhaseEvaluator
from libcity.evaluator.residuals_distribution import ResidualsDistribution
from libcity.model import loss

from analysis.MCTS_suggestions import MCTSSuggestions

from libcity.data.famvemd_2d import EMD2DmV
from scipy.signal import detrend
# from gevent import monkey
# monkey.patch_all()
import numbers
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# data and configs related global variables
task = 'air_quality_pred'
existed_task_data_model = None
cur_run_config = OrderedDict()
model_configs = OrderedDict()

dataset_infor = None
model_infor = OrderedDict()
cur_dataset_infor = None
cur_data_infor = OrderedDict()

dataset_timestamps = []
geo_coords = []
input_dataset_infor = OrderedDict()
raw_dataset = None
input_dataset = None
output_dataset = None

# phases, events related global variables
phases = OrderedDict()
entities = OrderedDict()
events = OrderedDict()

# truth and prediction related global variables
all_truth = OrderedDict()
all_prediction = OrderedDict()
all_residuals = OrderedDict()

truth_val_series = OrderedDict()
pred_val_series = OrderedDict()
residuals_series = OrderedDict()
# truth_flag_series = OrderedDict()
# pred_flag_series = OrderedDict()
# truth_level_series = OrderedDict()
# pred_level_series = OrderedDict()

# error indicator related global variables
step_err_infor_all = OrderedDict()
error_indicators = None
# all_residual_bins, all_mid_bins, all_pos_extreme_bins, all_neg_extreme_bins = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
all_residual_bins, all_mid_bins, all_pos_extreme_bins, all_neg_extreme_bins = [], [], [], []
all_residual_hists, all_mid_hists, all_pos_extreme_hists, all_neg_extreme_hists = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

# point metadata related global variables
metadata_raw = []
metadata_ins = []
metadata_series = []

metadata_objs = OrderedDict()
meta_attrs_level = OrderedDict()
sel_subsets_infor = OrderedDict()
ranges = None
slices = None
ranges_indices = None
slices_indices = None

def remove_duplicates(lst):
    return list(OrderedDict.fromkeys(lst))

@app.route("/grid_borders/<data_name>")
@cross_origin()
def get_grid_borders(data_name):
    # loc_ids = json.loads(loc_ids)
    # client = pymongo.MongoClient('mongodb://localhost:27017')
    # db = client.grid_aq_data
    # collection = db.grid_borders
    # data = collection.find({'loc_id': {'$in': loc_ids}}, {'_id':0})
    # grid_borders = list(data)
    
    grid_borders_file = f"./raw_data/{data_name}/grid_borders.json"
    with open(grid_borders_file, 'r', encoding='utf-8') as f:
        grid_borders = json.load(f)

    return jsonify(grid_borders)

@app.route("/task_data_model_infor")
@cross_origin()
def get_task_data_model_infor ():
    global existed_task_data_model
    global dataset_infor
    global model_infor
    
    error_configs = None
    indicator_schemes = None
    with open('./existed_task_data_model.json', 'r', encoding='utf-8') as f:
        existed_task_data_model = json.load(f)
    with open('./dataset_infor.json', 'r', encoding='utf-8') as f:
        dataset_infor = json.load(f)
    with open('./model_infor.json', 'r', encoding='utf-8') as f:
        model_infor = json.load(f)
    with open('./error_configs.json', 'r', encoding='utf-8') as f:
        error_configs = json.load(f)
        
    infor = {}
    infor['existed_task_data_model'] = existed_task_data_model
    infor['dataset_infor'] = dataset_infor
    infor['model_infor'] = model_infor
    infor['error_configs'] = error_configs

    return infor

@app.route("/run_configs/<config_file_name>")
@cross_origin()
def get_run_configs (config_file_name):
    global cur_run_config
    cur_run_config_file = f"./{config_file_name}.json"
    with open(cur_run_config_file, 'r', encoding='utf-8') as f:
        cur_run_config = json.load(f)
    return 'ok'

@app.route("/dataset_configs/<data_name>/<config_name>")
@cross_origin()
def get_dataset_configs (data_name, config_name):
    global cur_run_config
    cur_run_config_file = f"./{config_name}.json"
    with open(cur_run_config_file, 'r', encoding='utf-8') as f:
        cur_run_config = json.load(f)
    cur_run_config['dataset'] = data_name
    
    return {'ordered_data_str': json.dumps(cur_run_config)}

# get dataset information and raw dataset
@app.route("/cur_data_infor/<task>/<data_name>")
@cross_origin()
def get_cur_data_infor (task, data_name):
    # get dataset information
    global dataset_infor
    with open('./dataset_infor.json', 'r', encoding='utf-8') as f:
        dataset_infor = json.load(f)
    
    global cur_dataset_infor
    cur_dataset_infor = dataset_infor[data_name]
    time_range = cur_dataset_infor['Duration']
    
    global cur_run_config
    cur_run_config_file = f"./{cur_dataset_infor['Default_config']}.json"
    with open(cur_run_config_file, 'r', encoding='utf-8') as f:
        cur_run_config = json.load(f)
    cur_run_config['dataset'] = data_name
    
    # load raw data configs
    data_file_path = './raw_data/' + data_name + '/'
    dataset_path = './libcity/cache/dataset_cache/'
    ensure_dir(dataset_path)
    
    # get data features
    data_config_file = data_file_path + 'config.json'
    with open(data_config_file, 'r', encoding='utf-8') as f:
        data_config = json.load(f)
    features = list(data_config['dyna']['state'].keys())
    del features[0]
    out_dim = data_config['info'].get('output_dim', len(features))
    output_features = features[:out_dim]
    data_col = data_config['info'].get('data_col', features)
    
    dyna_file = data_file_path + data_name + '.dyna'
    date_format = "%Y%m-%d %H:%M:%S"
    # df = pd.read_csv(dyna_file, parse_dates=['time'])
    df = pd.read_csv(dyna_file, usecols=['time'], dtype={'time': str})
    global dataset_timestamps
    timestamps = df['time']
    dataset_timestamps = remove_duplicates(timestamps.tolist())
    # print('dataset_timestamps', dataset_timestamps)
    if task == 'air_quality_pred':
        raw_dataset_file_name = dataset_path + 'raw_air_quality_' + data_name + ".npy"

    # load binary raw dataset file
    global raw_dataset
    raw_dataset = np.load(raw_dataset_file_name)
    print("raw_dataset shape: ", raw_dataset.shape)
    
    timestamp_num = raw_dataset.shape[0]
    
    # compute value range of the dataset
    output_range = {}
    output_range['min'] = float(np.min(raw_dataset[:,:,0]))
    output_range['max'] = float(np.max(raw_dataset[:,:,0]))
    
    # get location coordinates
    global geo_coords
    geo_file = data_file_path + data_name + '.geo'
    df = pd.read_csv(geo_file)
    loc_ids = df['geo_id'].values.tolist()
    geo_coords = df['coordinates'].values.tolist()
    geo_coords = [json.loads(coord) for coord in geo_coords]
    # print('geo_coords: ', geo_coords)
    
    loc_list = df.to_dict('records')
    loc_x_list = []
    loc_y_list = []
    # loc_region_list = []
    for i in range(len(loc_list)):    
        loc_x_y = eval(loc_list[i]['coordinates'])
        loc_list[i]['coordinates'] = eval(loc_list[i]['coordinates'])
        loc_x_list.append(loc_x_y[0])
        loc_y_list.append(loc_x_y[1])
        
    loc_center_x = (max(loc_x_list) + min(loc_x_list)) / 2
    loc_center_y = (max(loc_y_list) + min(loc_y_list)) / 2
    loc_center = [loc_center_x, loc_center_y]
    
    # get grid border
    grid_borders_file = f"./raw_data/{data_name}/grid_borders.json"
    with open(grid_borders_file, 'r', encoding='utf-8') as f:
        grid_borders = json.load(f)
    # client = pymongo.MongoClient('mongodb://localhost:27017')
    # db = client.grid_aq_data
    # collection = db.grid_borders
    # data = collection.find({'loc_id': {'$in': loc_ids}}, {'_id':0})
    # grid_borders = list(data)
    
    global cur_data_infor
    cur_data_infor['default_config_file'] = cur_dataset_infor['Default_config']
    cur_data_infor['config_files'] = cur_dataset_infor['Config_files']
    cur_data_infor['space'] = {
        'loc_list': loc_list,
        'loc_center': loc_center,
        'grid_borders': grid_borders,
    }
    cur_data_infor['time'] = {
        'time_range': time_range,
        'time_num': timestamp_num,
    }
    cur_data_infor['features'] = {
        'num': len(data_col),
        'input': data_col,
        'output': output_features
    }
    cur_data_infor['threshold'] = {
        'output_range': output_range,
    }
    
    # cur_data_infor['raw_data'] = raw_dataset.tolist()

    return {'ordered_data_str': json.dumps(cur_data_infor)}

def generate_file_label(attributes):
    sorted_attributes = sorted(attributes)
    attributes_str = "_".join(sorted_attributes)
    file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
    file_label = f"{file_hash[:8]}"
    return file_label

@app.route("/input_output_data/<data_name>/<input_window>/<output_window>/<output_offset>")
@cross_origin()
def get_input_output_data (data_name, input_window, output_window, output_offset):
    input_window = int(input_window)
    output_window = int(output_window)
    output_offset = int(output_offset)
    
    global cur_data_infor
    cur_data_infor['time']['input_window'] = input_window
    cur_data_infor['time']['output_window'] = output_window
    cur_data_infor['time']['output_offset'] = output_offset
    
    # get input and output data
    global dataset_infor
    dataset_path = './libcity/cache/dataset_cache/'
    ensure_dir(dataset_path)    
    global cur_dataset_infor
    # input and output data are not normalized, not segmented to train/val/test
    input_dataset_file_name = f"{dataset_path}input_air_quality_{data_name}_{input_window}_{output_window}_{output_offset}.npy"
    output_dataset_file_name = f"{dataset_path}output_air_quality_{data_name}_{input_window}_{output_window}_{output_offset}.npy"

    # load input and output data
    global input_dataset, output_dataset
    input_dataset = np.load(input_dataset_file_name)
    output_dataset = np.load(output_dataset_file_name)
    
    # load point metadata
    global metadata_raw, metadata_ins, metadata_series
    attr_label = generate_file_label(cur_run_config['point_metadata'])
    metadata_file_raw = f'./meta_data/{data_name}/point_metadata_raw_{data_name}_{attr_label}.npz'
    metadata_file_ins = f'./meta_data/{data_name}/point_metadata_instances_{data_name}_{attr_label}_{input_window}_{output_window}_{output_offset}.npz'
    metadata_file_series = f'./meta_data/{data_name}/point_metadata_series_{data_name}_{attr_label}_{input_window}_{output_window}_{output_offset}.npz'
    metadata_raw = np.load(metadata_file_raw, allow_pickle=True)
    metadata_ins = np.load(metadata_file_ins, allow_pickle=True)
    metadata_series = np.load(metadata_file_series, allow_pickle=True)

    return 'done'

@app.route("/process_model_preds/<data_name>/<model_names>/<focus_th>/<focus_levels>")
@cross_origin()
def process_model_preds (data_name, model_names, focus_th, focus_levels):
    model_names = json.loads(model_names)
    focus_th = float(focus_th)
    focus_levels = json.loads(focus_levels)
    
    global all_truth, all_prediction, all_residuals
    global truth_val_series, pred_val_series, residuals_series
    global cur_run_config

    preds_num = None

    for model_name in model_names:
        if len(model_name) == 0: continue
        model_type, model_id = model_name.split('-')
        
        all_truth[model_name] = []
        all_prediction[model_name] = []
        all_residuals[model_name] = []
        
        # load model's predictions
        pred_processed_path = './analysis/prediction_processed/'
        ensure_dir(pred_processed_path)
        pred_processed_file = pred_processed_path + 'pred_truth_{}_{}_{}.npz'.format(model_name, focus_th, focus_levels)
        if os.path.exists(pred_processed_file):
            pred_processed = np.load(pred_processed_file)
            all_prediction[model_name] = pred_processed['all_prediction']
            all_truth[model_name] = pred_processed['all_truth']
            print(f'Load {pred_processed_file}')
        else:
            pred_truth_path = './libcity/cache/' + model_id + '/evaluate_cache/'
            ensure_dir(pred_truth_path)
            pred_truth_file_name_train = pred_truth_path + model_type + '_' + data_name + '_train_predictions.npz'
            pred_truth_file_name_valid = pred_truth_path + model_type + '_' + data_name + '_valid_predictions.npz'
            pred_truth_file_name_test = pred_truth_path + model_type + '_' + data_name + '_test_predictions.npz'
            train_results = np.load(pred_truth_file_name_train)
            valid_results = np.load(pred_truth_file_name_valid)
            test_results = np.load(pred_truth_file_name_test)
            all_prediction[model_name] = np.round(np.concatenate((train_results['prediction'], valid_results['prediction'], test_results['prediction']), axis=0)[:,:,:,0], decimals=2)
            all_truth[model_name] = np.round(np.concatenate((train_results['truth'], valid_results['truth'], test_results['truth']), axis=0)[:,:,:,0], decimals=2)
            all_prediction[model_name][all_prediction[model_name] < 0] = 0.000001
            np.savez(pred_processed_file,
                    all_truth=all_truth[model_name],
                    all_prediction=all_prediction[model_name])
            
        all_residuals[model_name] = all_prediction[model_name] - all_truth[model_name]
        preds_num = all_truth[model_name].size
        
        n_sample, window_size, n_loc = all_truth[model_name].shape
        error_step_num = raw_dataset.shape[0] - cur_run_config['input_window'] - cur_run_config['output_offset'] + 1
        # process model's predictions as step series
        truth_val_series[model_name] = []
        pred_val_series[model_name] = []
        residuals_series[model_name] = []
        
        str_bin_edges = list(map(str, focus_levels))
        step_infor_path = './analysis/data_step_infor/'
        ensure_dir(step_infor_path)
        step_infor_file = step_infor_path + 'step_infor_{}_{}_{}.npz'.format(model_name, focus_th, focus_levels)

        # if os.path.exists(step_infor_file):
        #     step_infor = np.load(step_infor_file, allow_pickle=True)
        #     step_truth_val = step_infor['step_truth_val']
        #     step_pred_val = step_infor['step_pred_val']
        #     step_residuals = step_infor['step_residuals']
        #     print(f'Load {step_infor_file}')
        # else:
        step_truth_val, step_pred_val, step_residuals = get_step_series(all_truth[model_name], all_prediction[model_name], all_residuals[model_name], cur_run_config['input_window'], cur_run_config['output_offset'])
        #     np.savez(step_infor_file, 
        #             step_truth_val=step_truth_val, 
        #             step_pred_val=step_pred_val,
        #             step_residuals=step_residuals)
        # step_truth_val, step_pred_val, step_residuals = step_truth_val.tolist(), step_pred_val.tolist(), step_residuals.tolist()
        # # 过滤掉np.nan，得到list
        # for i in range(len(step_truth_val)):
        #     if i > window_size and i < error_step_num - window_size: continue
        #     for j in range(len(step_truth_val[i])):
        #         filtered_lst = [x for x in step_truth_val[i][j] if not np.isnan(x)]
        #         step_truth_val[i][j] = filtered_lst
        #         filtered_lst = [x for x in step_pred_val[i][j] if not np.isnan(x)]
        #         step_pred_val[i][j] = filtered_lst
        #         filtered_lst = [x for x in step_residuals[i][j] if not np.isnan(x)]
        #         step_residuals[i][j] = filtered_lst
        
        truth_val_series[model_name], pred_val_series[model_name], residuals_series[model_name] = step_truth_val, step_pred_val, step_residuals
    
    return {'preds_num': preds_num}

# load and process models' predictions
@app.route("/model_parameters/<data_name>/<model_names>/<focused_model>/<baseline_model>")
@cross_origin()
def get_model_parameters (data_name, model_names, focused_model, baseline_model):
    model_names = json.loads(model_names)
    
    global model_configs
    model_configs = OrderedDict()
    for model_name in model_names:
        if len(model_name) == 0: continue
        model_type, model_id = model_name.split('-')
        # load model's config file
        model_config_file = f'./libcity/cache/{str(model_id)}/model_config.json'
        with open(model_config_file, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        model_configs[model_name] = model_config
    
    with open('./saved_subgroup_collections.json', 'r', encoding='utf-8') as f:
        saves_collections = json.load(f)
    
    used_model = None
    if baseline_model == 'none': used_model = focused_model
    else: used_model = baseline_model
    
    slice_indices_file = './point_slices/slice_indices.json'
    with open(slice_indices_file, 'r', encoding='utf-8') as f:
        slice_indices = json.load(f)
    
    cached_slices_configs = slice_indices[data_name][used_model]
    cur_slice_config = {
        "focus_th": cur_run_config['focus_th'],
        "range_params": cur_run_config['range_params'],
        "slice_params": cur_run_config['slice_params']
    }
    cached_slice_id = None
    for slice_id in cached_slices_configs:
        if cur_slice_config == cached_slices_configs[slice_id]:
            cached_slice_id = slice_id
            break
    cur_saved_collections = []
    if data_name in saves_collections:
        cur_key = f'{used_model}_{cached_slice_id}'
        if cur_key in saves_collections[data_name]:
            cur_saved_collections = saves_collections[data_name][cur_key]
    return {'model_configs': model_configs,
            'saves_collections': cur_saved_collections}
            


@app.route("/construct_subgroup/<data_name>/<model_names>/<configs>/<forecast_scopes>/<attr_bin_edges>/<select_ranges>")
@cross_origin()
def construct_subgroup (data_name, model_names, configs, forecast_scopes, attr_bin_edges, select_ranges):
    model_names = json.loads(model_names)
    configs = json.loads(configs)
    forecast_scopes = json.loads(forecast_scopes)
    attr_bin_edges = json.loads(attr_bin_edges)
    select_ranges = json.loads(select_ranges)
    
    # 获取相关的slice
    new_subset_strs = []
    select_indices = np.ones_like(metadata_ins[cur_run_config['point_metadata'][0]].flatten(), dtype=bool)
    for attr in select_ranges:
        if len(select_ranges[attr]) == 0: continue
        attr_indices = np.zeros_like(attr_flatten, dtype=bool)
        for range_str in select_ranges[attr]:
            new_subset_strs.append(range_str)
            cur_indices = ranges_indices[range_str]
            attr_indices = cur_indices | attr_indices
        select_indices = attr_indices & select_indices
    
    new_subgroup = {}
    # 计算两个模型的子集相关的属性 - size、error、三种残差分布
    for model_name in model_names:
        if len(model_name) == 0: continue
        cur_subgroup = {}
        cur_residuals = all_residuals[model_name][select_indices]
        cur_subgroup['sup_num'] = int(np.count_nonzero(select_indices))
        cur_subgroup['sup_rate'] = round(cur_subgroup['sup_num'] / residuals.size, 4)
        cur_subgroup['residual_abs'] = round(np.mean(np.abs(cur_residuals)), 4)
        cur_subgroup['subset_attrs'] = new_subset_strs
        
        cur_residual_hist, _ = np.histogram(cur_residuals, bins=all_residual_bins, density=False)
        cur_pos_extreme_hist, _ = np.histogram(cur_residuals, bins=all_pos_extreme_bins, density=False)
        cur_neg_extreme_hist, _ = np.histogram(cur_residuals, bins=all_neg_extreme_bins, density=False)
        cur_subgroup['residual_hist'] = cur_residual_hist.tolist()
        cur_subgroup['pos_extreme_hist'] = cur_pos_extreme_hist.tolist()
        cur_subgroup['neg_extreme_hist'] = cur_neg_extreme_hist.tolist()
        
        cur_residual_hist_sums = np.histogram(cur_residual, bins=all_residual_bins, weights=cur_residuals)[0]
        cur_pos_extreme_hist_sums = np.histogram(cur_residual, bins=all_pos_extreme_bins, weights=cur_residuals)[0]
        cur_neg_extreme_hist_sums = np.histogram(cur_residual, bins=all_neg_extreme_bins, weights=cur_residuals)[0]
        
        cur_subgroup['residual_hist_mean'] = np.divide(cur_residual_hist_sums, cur_residual_hist, where=(cur_residual_hist > 0)).tolist()
        cur_subgroup['pos_extreme_hist_mean'] = np.divide(cur_pos_extreme_hist_sums, cur_pos_extreme_hist, where=(cur_pos_extreme_hist > 0)).tolist()
        cur_subgroup['neg_extreme_hist_mean'] = np.divide(cur_neg_extreme_hist_sums, cur_neg_extreme_hist, where=(cur_neg_extreme_hist > 0)).tolist()
        
        cur_subgroup['residual_hist_normalize'] = np.round(cur_residual_hist / np.max(cur_residual_hist), 6).tolist()
        
        new_subgroup[model_name] = cur_subgroup
        
    return new_subgroup

@app.route("/attr_distributions/<data_name>/<model_names>/<configs>/<forecast_scopes>/<attr_bin_edges>/<select_ranges>")
@cross_origin()
def get_attr_distributions (data_name, model_names, configs, forecast_scopes, attr_bin_edges, select_ranges):
    model_names = json.loads(model_names)
    configs = json.loads(configs)
    forecast_scopes = json.loads(forecast_scopes)
    attr_bin_edges = json.loads(attr_bin_edges)
    select_ranges = json.loads(select_ranges)
    
    attr_distributions = OrderedDict()
    attr_flattens = OrderedDict()
    select_indices = np.ones_like(metadata_ins[cur_run_config['point_metadata'][0]].flatten(), dtype=bool)
    for attr in select_ranges:
        attr_flatten = np.array(flatten_list(metadata_series[attr]))
        attr_flattens[attr] = attr_flatten
        if len(select_ranges[attr]) == 0: continue
        attr_indices = np.zeros_like(attr_flatten, dtype=bool)
        for range_str in select_ranges[attr]:
            # bin = eval(range_str.split('=')[1])
            # cur_indices = (attr_flatten >= bin[0]) & (attr_flatten < bin[1])
            # range_strs = [range_infor['range_str'] for range_infor in ranges[attr]]
            # range_index = range_strs.index(range_str)
            # cur_indices = ranges[attr][range_index]['indices']
            cur_indices = ranges_indices[range_str]
            # print('series size', np.array(residuals_series[model_names[0]]).size)
            print('indices size', cur_indices.shape, attr_indices.shape)
            attr_indices = cur_indices | attr_indices
        select_indices = attr_indices & select_indices
    for attr in select_ranges:
        cur_attr_hist, _ = np.histogram(attr_flattens[attr][select_indices], bins=attr_bin_edges[attr], density=False)
        attr_distributions[attr] = cur_attr_hist.tolist()
    
    return attr_distributions

@app.route("/generate_subgroup/<data_name>/<baseline_model_name>/<focused_model_name>/<configs>/<forecast_scopes>/<attr_bin_edges>/<select_ranges>")
@cross_origin()
def generate_subgroup (data_name, baseline_model_name, focused_model_name, configs, forecast_scopes, attr_bin_edges, select_ranges):
    model_names = json.loads(model_names)
    configs = json.loads(configs)
    forecast_scopes = json.loads(forecast_scopes)
    attr_bin_edges = json.loads(attr_bin_edges)
    select_ranges = json.loads(select_ranges)
    
    select_indices = np.ones_like(metadata_ins[cur_run_config['point_metadata'][0]].flatten(), dtype=bool)
    for attr in select_ranges:
        if len(select_ranges[attr]) == 0: continue
        attr_indices = np.zeros_like(metadata_ins[cur_run_config['point_metadata'][0]].flatten(), dtype=bool)
        for range_str in select_ranges[attr]:
            cur_indices = ranges_indices[range_str]
            attr_indices = cur_indices | attr_indices
        select_indices = attr_indices & select_indices
    
    subgroup_model_name = ""
    if baseline_model_name == 'none': subgroup_model_name = focused_model_name
    else: subgroup_model_name = baseline_model_name
    residuals = all_residuals[subgroup_model_name]
    
    cur_subset_infor = {}
    cur_subset_infor['indices'] = select_indices
    cur_subset_error = residuals[cur_subset_infor['indices']]
    cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
    if np.mean(cur_subset_error) >= 0: cur_subset_infor['err_polarity'] = 'pos'
    else: cur_subset_infor['err_polarity'] = 'neg'
    
    for attr in select_ranges:
        if len(select_ranges[attr]) == 0: continue
        
    cur_subset_infor['subset_attrs'] = list(cur_subset)
    
    cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
    cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
    
        

@app.route("/attr_bin_errors/<data_name>/<model_names>/<subset_id>/<configs>/<forecast_scopes>")
@cross_origin()
def get_attr_bin_errors (data_name, model_names, subset_id, configs, forecast_scopes):
    model_names = json.loads(model_names)
    configs = json.loads(configs)
    forecast_scopes = json.loads(forecast_scopes)
    
    cur_subset_infor = OrderedDict()
    if '-' in str(subset_id):
        father_id = int(str(subset_id).split('-')[0]) - 1
        child_id = int(str(subset_id).split('-')[1]) - 1
        cur_subset_infor = slices[father_id]['contain_subsets'][child_id]
    else:
        cur_subset_infor = slices[int(subset_id)-1]
    
    attr_bin_errors = OrderedDict()
    hist_bin_errors = {
        'mid': [],
        'extreme_pos': [],
        'extreme_neg': []
    }
    limit_attrs = []

    cur_subset_indices = slices_indices[subset_id]
    if 'contain_subsets' not in cur_subset_infor:
        for range_str in cur_subset_infor['subset_attrs']:
            attr = range_str.split('=')[0]
            limit_attrs.append(attr)
    subset_residuals_flatten = OrderedDict()
    for model_name in model_names:
        if len(model_name) == 0: continue
        subset_residuals_flatten[model_name] = all_residuals[model_name].flatten()[cur_subset_indices]
    
    for attr in configs['point_metadata']:
        attr_bin_errors[attr] = []
        if attr in limit_attrs: continue
        attr_subset_flatten = metadata_ins[attr].flatten()[cur_subset_indices]
        for bin_range in metadata_objs[attr]['bins']:
            cur_range_indices = (attr_subset_flatten >= bin_range[0]) & (attr_subset_flatten < bin_range[1])
            bin_error = OrderedDict()
            for model_name in model_names:
                if len(model_name) == 0: continue
                if np.count_nonzero(cur_range_indices) == 0: bin_error[model_name] = 0
                else: bin_error[model_name] = np.mean(np.abs(subset_residuals_flatten[model_name][cur_range_indices]))
            attr_bin_errors[attr].append(bin_error)
    
    return attr_bin_errors

@app.route("/error_distributions/<data_name>/<model_names>/<configs>/<forecast_scopes>/<inspect_type>")
@cross_origin()
def get_error_distributions (data_name, model_names, configs, forecast_scopes, inspect_type):
    model_names = json.loads(model_names)
    configs = json.loads(configs)
    forecast_scopes = json.loads(forecast_scopes)
    boxes_infor = OrderedDict()
    global all_residual_bins, all_mid_bins, all_pos_extreme_bins, all_neg_extreme_bins
    tmp_residual_bins, tmp_mid_bins, tmp_pos_extreme_bins, tmp_neg_extreme_bins = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    model_num = 0
    for model_name in model_names:
        if len(model_name) == 0: continue
        model_num += 1
        err_dis_file = f"./error_distributions/{data_name}_{model_name}_{configs['distribution_params']['extreme_percentile']}_{forecast_scopes}.npz"
        print('err_dis_file', err_dis_file)
        if os.path.exists(err_dis_file):
            err_dis = np.load(err_dis_file, allow_pickle=True)
            residual_bins = err_dis['residual_bins']
            mid_bins = err_dis['mid_bins']
            pos_extreme_bins = err_dis['pos_extreme_bins']
            neg_extreme_bins = err_dis['neg_extreme_bins']
            
            residual_hists = err_dis['residual_hists']
            mid_hists = err_dis['mid_hists']
            pos_extreme_hists = err_dis['pos_extreme_hists']
            neg_extreme_hists = err_dis['neg_extreme_hists']
            
            tmp_residual_bins[model_name], tmp_mid_bins[model_name], tmp_pos_extreme_bins[model_name], tmp_neg_extreme_bins[model_name] = residual_bins.tolist(), mid_bins.tolist(), pos_extreme_bins.tolist(), neg_extreme_bins.tolist()
            all_residual_hists[model_name], all_mid_hists[model_name], all_pos_extreme_hists[model_name], all_neg_extreme_hists[model_name] = residual_hists.tolist(), mid_hists.tolist(), pos_extreme_hists.tolist(), neg_extreme_hists.tolist()
            
            boxes_infor[model_name] = err_dis['boxes_infor'].tolist()
        else:
            residual_dis_cls = ResidualsDistribution(configs, all_residuals[model_name], forecast_scopes)
            residual_bins_obj = residual_dis_cls.compute_residual_bins()
            tmp_residual_bins[model_name], tmp_mid_bins[model_name], tmp_pos_extreme_bins[model_name], tmp_neg_extreme_bins[model_name] = residual_bins_obj['residual_bins'], residual_bins_obj['mid_bins'], residual_bins_obj['pos_extreme_bins'], residual_bins_obj['neg_extreme_bins']
            boxes_infor[model_name] = residual_dis_cls.compute_residual_boxes()
            residual_hists_obj = residual_dis_cls.compute_residual_distribution()
            all_residual_hists[model_name], all_mid_hists[model_name], all_pos_extreme_hists[model_name], all_neg_extreme_hists[model_name] = residual_hists_obj['residual_hists'], residual_hists_obj['mid_hists'], residual_hists_obj['pos_extreme_hists'], residual_hists_obj['neg_extreme_hists']
            residual_dis_cls.save_residual_distributions(model_name)
        
    if model_num > 1:
        residual_bins_list = list(tmp_residual_bins.values())
        common_min = min([residual_bins_list[0][0], residual_bins_list[1][0]])
        common_max = max([residual_bins_list[0][-1], residual_bins_list[1][-1]])
        common_neg_limit = min([residual_bins_list[0][1], residual_bins_list[1][1]])
        common_pos_limit = max([residual_bins_list[0][-2], residual_bins_list[1][-2]])
        common_neg_in_limit = min([residual_bins_list[0][2], residual_bins_list[1][2]])
        common_pos_in_limit = max([residual_bins_list[0][-3], residual_bins_list[1][-3]])
        
        # 根据in_limit生成mid_bin_edges
        step_len = configs['distribution_params']['val_step']
        common_mid_bins = [common_neg_limit]
        for bin_edge in range(common_neg_in_limit, common_pos_in_limit+1, step_len):
            common_mid_bins.append(bin_edge)
        common_mid_bins.append(common_pos_limit)
        common_whole_bins = [common_min]
        common_whole_bins += common_mid_bins
        common_whole_bins.append(common_max)
        # 计算两个模型的在common bins中的hist
        common_neg_outlier_hist = OrderedDict()
        common_pos_outlier_hist = OrderedDict()
        common_mid_bin_hist = OrderedDict()
        common_residual_bin_hist = OrderedDict()
        
        residual_dis_cls = ResidualsDistribution(configs, None, forecast_scopes)
        
        extreme_pos_residuals = np.concatenate((all_residuals[model_names[0]][all_residuals[model_names[0]] > common_pos_limit], all_residuals[model_names[1]][all_residuals[model_names[1]] > common_pos_limit]))
        extreme_neg_residuals = np.concatenate((all_residuals[model_names[0]][all_residuals[model_names[0]]  < common_neg_limit], all_residuals[model_names[1]][all_residuals[model_names[1]] < common_neg_limit]))
        
        common_neg_outlier_bins = residual_dis_cls.divide_extreme_range(extreme_neg_residuals)
        common_pos_outlier_bins = residual_dis_cls.divide_extreme_range(extreme_pos_residuals)
        
        for model_name in model_names:
            common_hists = residual_dis_cls.compute_residual_distribution(all_residuals[model_name], common_whole_bins, common_mid_bins, common_pos_outlier_bins, common_neg_outlier_bins, None)
            common_residual_bin_hist[model_name], common_mid_bin_hist[model_name], common_neg_outlier_hist[model_name], common_pos_outlier_hist[model_name] = common_hists['residual_hists'], common_hists['mid_hists'], common_hists['neg_extreme_hists'], common_hists['pos_extreme_hists']
            boxes_infor[model_name] = residual_dis_cls.compute_residual_boxes(all_residuals[model_name], common_whole_bins, forecast_scopes)
        
        all_residual_bins = common_whole_bins
        all_mid_bins = common_mid_bins
        all_neg_extreme_bins = common_neg_outlier_bins
        all_pos_extreme_bins = common_pos_outlier_bins
        
        return  {'ordered_data_str': json.dumps({
            'all_residual_bins': common_whole_bins,
            'all_mid_bins': common_mid_bins,
            'all_neg_extreme_bins': common_neg_outlier_bins,
            'all_pos_extreme_bins': common_pos_outlier_bins,
            'all_residual_hists': common_residual_bin_hist,
            'all_neg_extreme_hists': common_neg_outlier_hist,
            'all_pos_extreme_hists': common_pos_outlier_hist,
            'all_mid_hists': common_mid_bin_hist,
            'boxes_infor': boxes_infor
        })}
    else:
        for model_name in model_names:
            if len(model_name) == 0: continue
            all_residual_bins = tmp_residual_bins[model_name]
            all_mid_bins = tmp_mid_bins[model_name]
            all_neg_extreme_bins = tmp_neg_extreme_bins[model_name]
            all_pos_extreme_bins = tmp_pos_extreme_bins[model_name]

        return {'ordered_data_str': json.dumps({
            'all_residual_bins': all_residual_bins,
            'all_mid_bins': all_mid_bins,
            'all_pos_extreme_bins': all_pos_extreme_bins,
            'all_neg_extreme_bins': all_neg_extreme_bins,
            'all_residual_hists': all_residual_hists,
            'all_mid_hists': all_mid_hists,
            'all_pos_extreme_hists': all_pos_extreme_hists,
            'all_neg_extreme_hists': all_neg_extreme_hists,
            'boxes_infor': boxes_infor
        })}

def find_ndarray_keys(d, parent_key=''):
    """
    递归遍历字典，找到值为ndarray类型的键，并返回这些键的完整路径。
    """
    ndarray_keys = []
    
    if isinstance(d, dict):
        for key, value in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, np.ndarray):
                ndarray_keys.append(full_key)
            elif isinstance(value, dict):
                ndarray_keys.extend(find_ndarray_keys(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        ndarray_keys.extend(find_ndarray_keys(item, f"{full_key}[{i}]"))
    
    return ndarray_keys

@app.route("/st_phase_events/<data_name>/<model_names>/<focus_th>/<phase_params>/<bin_list>/<event_params>/<forecast_scopes>")
@cross_origin()
def get_st_phase_events (data_name, model_names, focus_th, phase_params, bin_list, event_params, forecast_scopes):
    focus_th = float(focus_th)
    phase_params = json.loads(phase_params, object_pairs_hook=OrderedDict)
    max_gap_len = phase_params['max_gap_len']
    min_length = phase_params['min_len']
    bin_list = json.loads(bin_list)
    event_params = json.loads(event_params, object_pairs_hook=OrderedDict)
    model_names = json.loads(model_names)
    forecast_scopes = json.loads(forecast_scopes)
    
    phases_attr_bin_edges = OrderedDict()
    
    global phases, entities, events
    for model_name in model_names:
        if len(model_name) == 0: continue
        params_str = generate_file_label(cur_run_config['phase_params']['attributes'])
        phases_file = f'./phases_data/{data_name}_{model_name}_{focus_th}_{min_length}_{max_gap_len}_{params_str}.npy'
        phases_attr_bin_file = f'./phases_data/{data_name}_{model_name}_{focus_th}_{min_length}_{max_gap_len}_{params_str}.json'
        if os.path.exists(phases_file):
            phases[model_name] = np.load(phases_file, allow_pickle=True).tolist()
            with open(phases_attr_bin_file, 'r', encoding='utf-8') as f:
                phases_attr_bin_edges[model_name] = json.load(f)
            # print(phases_attr_bin_edges[model_name])
        else:
            phases_cls = Phases(cur_run_config, raw_dataset, model_name, dataset_timestamps, np.array(geo_coords), phase_params)
            phases_cls.get_all_phases()
            residual_bins_obj = {
                'residual_bins': all_residual_bins,
                'mid_bins': all_mid_bins,
                'pos_extreme_bins': all_pos_extreme_bins,
                'neg_extreme_bins': all_neg_extreme_bins
            }
            phases_eval_cls = PhaseEvaluator(raw_dataset, model_name, cur_run_config, phases, truth_val_series, pred_val_series, residuals_series, residual_bins_obj, forecast_scopes)
            phases_eval_cls.compute_phases_err_metrics()
            phases_attr_bin_edges[model_name] = phases_eval_cls.compute_phases_attr_metric_bins()
            phases[model_name] = phases_cls.all_phases
            phases_cls.save_phases()
    
    params_str = '_'.join([str(value) for value in cur_run_config['event_params'].values() if not isinstance(value, list)])
    attr_str = '_'.join(sorted(cur_run_config['event_params']['attributes']))
    metric_str = '_'.join(sorted(cur_run_config['event_params']['error_metrics']))
    attributes_str = params_str + '_' + attr_str + '_' + metric_str
    file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
    file_label = f"{file_hash[:8]}"
        
    entities_events_file = f"./entities_events/{data_name}_{float(focus_th)}_{file_label}.npz"
    
    # event_params_str = '_'.join([str(val) for val in event_params.values()])
    # entities_events_file = f"./entities_events/{data_name}_{focus_th}_{event_params_str}.npz"
    
    # load entities and events
    if os.path.exists(entities_events_file):
        entities_events = np.load(entities_events_file, allow_pickle=True)
        # entities[model_name] = entities_events['entities'].tolist()
        # events[model_name] = entities_events['events'].tolist()
        entities = entities_events['entities'].tolist()
        events = entities_events['events'].tolist()
    else:
        entities_events_cls = Entities_Events(cur_run_config, raw_dataset, phases, geo_coords, event_params)
        entities_events_cls.get_all_entities_events()
        # entities[model_name] = entities_events_cls.all_entities
        # events[model_name] = entities_events_cls.all_events
        entities = entities_events_cls.all_entities
        events = entities_events_cls.all_events
    # for i, entity in enumerate(entities):
    #     for j, en in enumerate(entity):
    #         del entity['data']
    # print(list(phases.keys()), list(entities.keys()), list(events.keys()))
    print('phases', phases[model_names[0]][0])
    
    phases_infor = {
        "phases": phases,
        "phases_attr_bin_edges": phases_attr_bin_edges,
        "entities": entities,
        "events": events,
        "time_strs": dataset_timestamps,
    }
    
    return phases_infor

# 这里先不考虑啦！先做调通子集
@app.route("/metadata_distribution/<data_name>/<model_name>/<subset_scope>/<phase_id>/<meta_attrs>/<fore_step>/<range_mode>/<range_params>/<subset_params>/<organize_params>/<val_bins>/<forecast_scopes>")
@cross_origin()
def get_metadata_distribution(data_name, model_name, subset_scope, phase_id, meta_attrs, fore_step, range_mode, range_params, subset_params, organize_params, val_bins, forecast_scopes):
    meta_attrs = json.loads(meta_attrs)
    global cur_run_config
    input_window = cur_run_config['input_window']
    output_window = cur_run_config['output_window']
    output_offset = cur_run_config['output_offset']
    global metadata_raw, metadata_ins, metadata_series
    # global all_truth, all_prediction
    residuals = all_prediction - all_truth
    
    # 统一加载一样的ranges？？？这样简单太多！否则每个scope都要加载也不好比较！
    
    # 这里可以支持不同scope、不同模型的对比，提取所有的scope结果，默认展示每个attr的第一个scope结果？
    # 先计算每个模型的每个scope的结果
    
        
    
    return 

def ensure_list_in_dict(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            d[key] = ensure_list_in_dict(value)
    return d

# @app.route("/subgroup_path/<data_name>/<model_name>/<configs>/<root_id>")
# @cross_origin()
# def get_subgroup_path(data_name, model_name, configs, root_id):

@app.route("/slices_indices/<data_name>/<model_name>/<cached_slice_id>")
@cross_origin() 
def get_slices_indices(data_name, model_name, cached_slice_id):
    cached_slice_id = int(cached_slice_id)
    
    global ranges_indices, slices_indices
    if ranges_indices is None:
        indices_file = f"./point_slices/indices_{data_name}_{model_name}_{cached_slice_id}.npz"
        ranges_slices_indices = np.load(indices_file, allow_pickle=True)
        ranges_indices = ranges_slices_indices['ranges_indices'].item()
        slices_indices = ranges_slices_indices['slices_indices'].item()
    
    return 'ok'

@app.route("/find_point_slices/<data_name>/<baseline_model_name>/<focused_model_name>/<configs>/<forecast_scopes>")
@cross_origin() 
def find_point_slices(data_name, baseline_model_name, focused_model_name, configs, forecast_scopes):
    configs = json.loads(configs)
    range_params = configs['range_params']
    subset_params = configs['slice_params']
    val_bins = all_residual_bins
    forecast_scopes = json.loads(forecast_scopes)
    subset_scope = subset_params['data_scope']
    
    global phases, error_indicators
    global all_truth, all_prediction, truth_val_series, pred_val_series
    global metadata_objs, ranges, slices, ranges_indices, slices_indices
    
    subgroup_model_name = ""
    if baseline_model_name == 'none': subgroup_model_name = focused_model_name
    else: subgroup_model_name = baseline_model_name
    
    
    # 判断是否存在缓存的slice结果
    # 读取slice indices文件
    slice_indices_file = './point_slices/slice_indices.json'
    with open(slice_indices_file, 'r', encoding='utf-8') as f:
        slice_indices = json.load(f)
    cached_slices_configs = slice_indices[data_name][subgroup_model_name]
    cur_slice_config = {
        "focus_th": cur_run_config['focus_th'],
        "range_params": range_params,
        "slice_params": subset_params
    }
    # print('cur_slice_config', cur_slice_config)
    cached_slice_id = None
    ranges = None
    slices = None
    ranges_indices = None
    slices_indices = None
    for slice_id in cached_slices_configs:
        if cur_slice_config == cached_slices_configs[slice_id]:
            cached_slice_id = slice_id
            break
    # print('cached_slice_id', cached_slice_id)
    if cached_slice_id is not None:
        ranges_slices_file = f"./point_slices/{data_name}_{subgroup_model_name}_{cached_slice_id}.npz"
        # slices_indices_file = f"./point_slices/indices_{data_name}_{model_name}_{cached_slice_id}.npz"
        ranges_slices = np.load(ranges_slices_file, allow_pickle=True)
        metadata_objs = ranges_slices['metadata_objs'].item()
        ensure_list_in_dict(metadata_objs)
        ranges = ranges_slices['ranges'].item()
        slices = ranges_slices['slices'].item()['slices']
        
        indices_file = f"./point_slices/indices_{data_name}_{subgroup_model_name}_{cached_slice_id}.npz"
        ranges_slices_indices = np.load(indices_file, allow_pickle=True)
        ranges_indices = ranges_slices_indices['ranges_indices'].item()
        slices_indices = ranges_slices_indices['slices_indices'].item()
    else:
        slices_cls = SliceEvaluator(configs, subset_scope, metadata_series, residuals_series[subgroup_model_name], val_bins, phases, forecast_scopes)
        slices_cls.subgroup_mining()
        metadata_objs = slices_cls.metadata_ranges
        ensure_list_in_dict(metadata_objs)
        ranges = slices_cls.ranges_infor
        slices = slices_cls.slices_infor['slices']
        ranges_indices = slices_cls.ranges_indices
        slices_indices = slices_cls.slices_indices
        slices_cls.save_ranges_slices()
    
    
    # 基于baseline model的range和slice信息，计算focused model的误差
    if len(baseline_model_name) > 0:
        focused_res_series = np.array(flatten_list(residuals_series[focused_model_name]))
        for attr in ranges:
            for range_infor in ranges[attr]:
                focused_res_abs = np.mean(np.abs(focused_res_series[ranges_indices[range_infor['range_str']]]))
                range_infor['abs_residual_comp'] = focused_res_abs
        for slice_infor in slices:
            cur_focused_res = focused_res_series[slices_indices[slice_infor['subset_id']]]
            focused_res_abs = np.mean(np.abs(cur_focused_res))
            cur_residual_hist, _ = np.histogram(cur_focused_res, bins=all_residual_bins, density=False)
            cur_pos_extreme_hist, _ = np.histogram(cur_focused_res, bins=all_pos_extreme_bins, density=False)
            cur_neg_extreme_hist, _ = np.histogram(cur_focused_res, bins=all_neg_extreme_bins, density=False)
            
            slice_infor['residual_abs_comp'] = round(focused_res_abs, 4)
            slice_infor['residual_hist_comp'] = cur_residual_hist.tolist()
            slice_infor['pos_extreme_hist_comp'] = cur_pos_extreme_hist.tolist()
            slice_infor['neg_extreme_hist_comp'] = cur_neg_extreme_hist.tolist()

            cur_residual_hist_sums = np.histogram(cur_focused_res, bins=all_residual_bins, weights=cur_focused_res)[0]
            cur_pos_extreme_hist_sums = np.histogram(cur_focused_res, bins=all_pos_extreme_bins, weights=cur_focused_res)[0]
            cur_neg_extreme_hist_sums = np.histogram(cur_focused_res, bins=all_neg_extreme_bins, weights=cur_focused_res)[0]
            
            slice_infor['residual_hist_mean'] = np.divide(cur_residual_hist_sums, cur_residual_hist, where=(cur_residual_hist > 0)).tolist()
            slice_infor['pos_extreme_hist_mean'] = np.divide(cur_pos_extreme_hist_sums, cur_pos_extreme_hist, where=(cur_pos_extreme_hist > 0)).tolist()
            slice_infor['neg_extreme_hist_mean'] = np.divide(cur_neg_extreme_hist_sums, cur_neg_extreme_hist, where=(cur_neg_extreme_hist > 0)).tolist()
            
    print('subset and range ready!')
    
    residual_abs_mean = 0
    
    return {
        "subset_scope": subset_scope,
        "subsets": slices,
        "range_infor": ranges,
        "meta_attr_objs": metadata_objs,
        "cached_slice_id": cached_slice_id,
        "clean_residual_abs_mean": round(residual_abs_mean, 2)
        }

def series_index_to_ins_index(series_index):
    time_id, space_id, step_id = tuple(series_index)
    sample_num = next(iter(all_truth.values())).shape[0]
    if time_id > sample_num:
        step_id += (time_id - sample_num)
        sample_id = time_id - step_id
    else:
        sample_id = time_id - step_id
    return [sample_id, step_id, space_id]

def add_gaussian_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def mixup_data(x1, y1, x2, y2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = np.array(lam, dtype=np.float32)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

@app.route("/data_augmentation_by_slices/<data_name>/<model_name>/<sel_subsets>/<focus_conditions>/<aug_params>")
@cross_origin()
def data_augmentation_by_slices(data_name, model_name, sel_subsets, focus_conditions, aug_params):
    sel_subsets = json.loads(sel_subsets)
    focus_conditions = json.loads(focus_conditions)
    aug_params = json.loads(aug_params)
    
    print('start data augmentation')
    # get original train, validation and test dataset
    train_rate = model_configs[model_name]['train_rate']
    eval_rate = model_configs[model_name]['eval_rate']
    test_rate = 1 - train_rate - eval_rate
    num_samples = input_dataset.shape[0]
    num_test = round(num_samples * test_rate)
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train
    
    all_truth_array = next(iter(all_truth.values()))
    original_x_train = input_dataset[:num_train]
    original_y_train = output_dataset[:num_train]
    x_train = input_dataset[:num_train]
    y_train = output_dataset[:num_train]
    x_val = input_dataset[num_train: num_train + num_val]
    y_val = output_dataset[num_train: num_train + num_val]
    x_test = input_dataset[-num_test:]
    y_test = output_dataset[-num_test:]
    metadata_train = OrderedDict()
    for attr in metadata_ins:
        metadata_train[attr] = metadata_ins[attr][:num_train]
    
    # 从子集id获取到子集信息，然后获取相应的st_index，筛选出相应的时间步
    # 如果subset是复合subset，则将每个包括的子subset插入；如果不是复合的，直接append即可
    # 也可能选择的是子subset
    sel_subsets_infor = []
    for subset_id in sel_subsets:
        if '-' in str(subset_id):
            father_id = int(str(subset_id).split('-')[0]) - 1
            child_id = int(str(subset_id).split('-')[1]) - 1
            cur_subset_infor = slices[father_id]['contain_subsets'][child_id]
            sel_subsets_infor.append(cur_subset_infor)
        else:
            sel_subsets_infor.append(slices[int(subset_id)-1])
    
    all_focus_conditions = []
    for subset_infor in sel_subsets_infor:
        cur_condition = subset_infor['range_val']
        all_focus_conditions.append(cur_condition)
    
    # 直接根据选择的subgroup,汇总st_indices,保存group和weight文件
    focus_flags_train = np.zeros_like(y_train[...,0], dtype=bool)
    for condition in all_focus_conditions:
        for attr in condition:
            val_range = condition[attr]
            cur_focus_flags = (metadata_train[attr] >= val_range[0]) & (metadata_train[attr] <= val_range[0])
            focus_flags_train = focus_flags_train | cur_focus_flags
    all_focus_st_indices = np.argwhere(focus_flags_train).tolist()
    all_sample_ids = []
    for index in all_focus_st_indices:
        ins_index = series_index_to_ins_index(index)
        all_sample_ids.append(ins_index[0])
    all_sample_ids = remove_duplicates(all_sample_ids)
    
    # 获取与关注自己相关的训练集
    focus_train_ins_ids = [id for id in all_sample_ids if id < original_x_train.shape[0]]
    x_train_focus = original_x_train[focus_train_ins_ids, ...]
    y_train_focus = original_y_train[focus_train_ins_ids, ...]
    x_train_unfocus = np.delete(original_x_train, focus_train_ins_ids, axis=0)
    y_train_unfocus = np.delete(original_y_train, focus_train_ins_ids, axis=0)
    
    # 在server部分进行数据增强，并处理文件！
    # 处理顺序：EMD增强->Mixup->IW或Group DRO
    new_instances = []
    new_ins_metadata = OrderedDict()
    if aug_params['EMD'] == 'MEMD':
        emd_param = {
            'nimfs': aug_params['imf_num'],  # 5
            'tol': aug_params['tol'],  # 0.05
            'type': 6,
            'plot': 'off'
        }
        all_join_instances = []
        all_IMFs = []
        new_x_train = []
        new_y_train = []
        
        for i in range(x_train_focus.shape[0]):
            instance = np.concatenate((x_train_focus[i], y_train_focus[i]), axis=0)            
            instance = np.transpose(instance, (2, 1, 0))
            all_join_instances.append(instance)
            result = None
            result = EMD2DmV(instance, emd_param)
            if result != 'fail':
                all_IMFs.append(result['IMF'])
                new_instance = np.zeros(all_join_instances[0].shape)
                for j in range(aug_params['EMD_rounds']):
                    random_vector = np.random.uniform(0, 2, size=aug_params['imf_num'])
                    new_instance = np.tensordot(result['IMF'], random_vector, axes=([3], [0]))
                    new_instance = np.transpose(new_instance, (2, 1, 0))
                    new_x_train.append(new_instance[:cur_run_config['input_window'],:,:])
                    new_y_train.append(new_instance[cur_run_config['input_window']:,:,:])
                    new_instances.append(new_instance)
            else:
                print("EMED error")
        
        # 令x_train和y_train是原始的+增强的
        new_x_train = np.stack(new_x_train)
        new_y_train = np.stack(new_y_train)
        x_train = np.concatenate([original_x_train, new_x_train])
        y_train = np.concatenate([original_y_train, new_y_train])
        # EMD增强数据后，新增加的样本都算是focus？- 可！
        x_train_focus = np.concatenate([x_train_focus, new_x_train])
        y_train_focus = np.concatenate([y_train_focus, new_y_train])
    
    if aug_params['mixup'] == 'balanced_mixup':
        # balanced_mixup分两步：1.从两组中确定一个；2.从组中选择样本
        mixup_num = x_train.shape[0] * aug_params['mixup_rate']
        mixed_x_train = []
        mixed_y_train = []
        for i in range(mixup_num):
            x1, x2, y1, y2 = None, None, None, None
            pool_x_1, pool_y_1, pool_x_2, pool_y_2 = None, None, None, None
            if np.random.rand() > 0.5:
                pool_x_1 = x_train_focus
                pool_y_1 = y_train_focus
            else:
                pool_x_1 = x_train_unfocus
                pool_y_1 = y_train_unfocus
            if np.random.rand() > 0.5:
                pool_x_2 = x_train_focus
                pool_y_2 = y_train_focus
            else:
                pool_x_2 = x_train_unfocus
                pool_y_2 = y_train_unfocus
            sample_id_1 = random.randint(0, pool_x_1.shape[0]-1)
            sample_id_2 = random.randint(0, pool_x_2.shape[0]-1)
            x1, y1 = pool_x_1[sample_id_1], pool_y_1[sample_id_1]
            x2, y2 = pool_x_2[sample_id_2], pool_y_2[sample_id_2]
            mixed_x, mixed_y = mixup_data(x1, y1, x2, y2, alpha=1.0)
            mixed_x_train.append(mixed_x)
            mixed_y_train.append(mixed_y)
            new_instance = np.concatenate((mixed_x, mixed_y), axis=0)
            new_instances.append(new_instance)
            
        mixed_x_train = np.stack(mixed_x_train)
        mixed_y_train = np.stack(mixed_y_train)
        x_train = np.concatenate([x_train, mixed_x_train])
        y_train = np.concatenate([y_train, mixed_y_train])
        
    # compute metadata for new instance
    for attr in cur_run_config['point_metadata']:
        new_ins_metadata[attr] = np.zeros((len(new_instances), cur_run_config['output_window'], raw_dataset.shape[1]))
    for i, new_instance in enumerate(new_instances):
        point_metadata = PointMetaData(model_configs[model_name], new_instance)
        point_metadata.get_metadata_by_type()
        for attr, vals in point_metadata.raw_meta_data.items():
            new_ins_metadata[attr][i] = vals[-cur_run_config['output_window']:]
    for attr in new_ins_metadata:
        print('shape:', metadata_train[attr].shape, new_ins_metadata[attr].shape)
        metadata_train[attr] = np.concatenate([metadata_train[attr], new_ins_metadata[attr]])
    
    # save as .npz file
    if aug_params['mixup'] is not None or aug_params['EMD'] is not None:
        aug_parameters_str = data_name + '_' + str(cur_run_config['input_window']) + '_' + str(cur_run_config['output_window']) + '_' + str(cur_run_config['train_rate']) + '_' + str(cur_run_config['eval_rate']) + '_' + str(cur_run_config['scaler']) + '_' + aug_params['mixup'] + '_' + aug_params['EMD'] + '_' + str(aug_params['imf_num']) + '_' + str(aug_params['EMD_rounds']) + '_' + str(aug_params['mixup_rate']) + '_' + str(aug_params['tol'])
        aug_file_name = os.path.join('./libcity/cache/dataset_cache/','aug_air_quality_{}.npz'.format(aug_parameters_str))
        np.savez_compressed(
            aug_file_name,
            original_x_train = original_x_train,
            original_y_train = original_y_train,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_val=x_val,
            y_val=y_val,
        )
        print('Augmented Data Saved at ' + aug_file_name)
    print('ins num after aug: ', x_train.shape[0], original_x_train.shape[0])

    # 根据增强的数据，计算相应的权重和组标签
    # 基于所选择子集的全部筛选条件，来算增强数据中具体哪些点是关注的
    # 需要重新计算元数据啊，只对新的训练集做
    # train_size = train_rate * all_truth_array.size
    train_size = y_train.size
    all_ins_num = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    group_labels = np.zeros((all_ins_num, all_truth_array.shape[1], all_truth_array.shape[2]))
    weights = np.ones((all_ins_num, all_truth_array.shape[1], all_truth_array.shape[2]))
    # if data are augmented, compute metadata for augmented data
    focus_flags_train = np.zeros_like(y_train[...,0], dtype=bool)
    for condition in all_focus_conditions:
        for attr in condition:
            val_range = condition[attr]
            cur_focus_flags = (metadata_train[attr] >= val_range[0]) & (metadata_train[attr] <= val_range[0])
            focus_flags_train = focus_flags_train | cur_focus_flags
    focus_train_size = np.count_nonzero(focus_flags_train)
    un_focus_train_size = train_size - focus_train_size
    focus_weight = train_size / (2 * focus_train_size)
    un_focus_weight = train_size / (2 * un_focus_train_size)
    group_labels[:y_train.shape[0]] = focus_flags_train.astype(int)
    weights[:y_train.shape[0]] = np.where(focus_flags_train, focus_weight, un_focus_weight)
    print('focus_weight', focus_weight)
    print('un_focus_weight', un_focus_weight)

    # 将group_labels和weights保存为文件
    exp_id = int(random.SystemRandom().random() * 100000)
    group_file = f"./libcity/group_files/group-{data_name}-{str(exp_id)}.npy"
    weight_file = f"./libcity/weight_files/weight-{data_name}-{str(exp_id)}.npy"
    np.save(group_file, group_labels)
    np.save(weight_file, weights)
    
    return 'ok'

# def get_step_series(truth, pred, residuals, input_window, output_offset):
#     n_sample, window_size, n_loc = truth.shape
    
#     error_step_num = raw_dataset.shape[0] - input_window - output_offset + 1

#     # 使用 NaN 初始化，避免未赋值的地方残留 0 值
#     truth_val_list = np.full((error_step_num, n_loc, window_size), np.nan)
#     pred_val_list = np.full((error_step_num, n_loc, window_size), np.nan)
#     residuals_list = np.full((error_step_num, n_loc, window_size), np.nan)

#     for i in range(n_sample):
#         for k in range(window_size):
#             truth_val_list[i:i+window_size, :, k] = truth[i, k, :]
#             pred_val_list[i:i+window_size, :, k] = pred[i, k, :]
#             residuals_list[i:i+window_size, :, k] = residuals[i, k, :]

#     # 反转最后一个维度的值
#     truth_val_list = truth_val_list[:, :, ::-1]
#     pred_val_list = pred_val_list[:, :, ::-1]
#     residuals_list = residuals_list[:, :, ::-1]

#     return truth_val_list, pred_val_list, residuals_list

def get_step_series(truth, pred, residuals, input_window, output_offset):
    n_sample, window_size, n_loc = truth.shape
    error_step_num = raw_dataset.shape[0] - input_window - output_offset + 1
    
    truth_val_list = []
    pred_val_list = []
    residuals_list = []
    for i in range(error_step_num):
        truth_val_list.append([])
        pred_val_list.append([])
        residuals_list.append([])
        for j in range(n_loc):
            truth_val_list[i].append([])
            pred_val_list[i].append([])
            residuals_list[i].append([])

    for i in range(n_sample):
        for k in range(window_size):
            for l in range(n_loc):
                truth_val_list[i+k][l].append(truth[i][k][l])
                pred_val_list[i+k][l].append(pred[i][k][l])
                residuals_list[i+k][l].append(residuals[i][k][l])
                
    for i in range(len(truth_val_list)):
        for j in range(len(truth_val_list[0])):
            truth_val_list[i][j].reverse()
            pred_val_list[i][j].reverse()
            residuals_list[i][j].reverse()
            
    return truth_val_list, pred_val_list, residuals_list

# 使用递归将三维列表变为一维
# def flatten_list(lst):
#     flat_list = []
#     for sublist in lst:
#         if isinstance(sublist, list):
#             flat_list.extend(flatten_list(sublist))
#         else:
#             flat_list.append(sublist)
#     return flat_list

def flatten_list(multi_list):
    return np.array([item for sublist1 in multi_list for sublist2 in sublist1 for item in sublist2])

def get_space_list(lst):
    space_list = []
    for i in range(len(lst[0])):
        space_list.append([])
        for j in range(len(lst)):
            space_list[i].extend(lst[j][i])
    return space_list

def get_time_list(lst):
    time_list = []
    for i in range(len(lst)):
        time_list.append([])
        for j in range(len(lst[i])):
            time_list[i].extend(lst[i][j])
    return time_list

def sum_of_nested_list(nested_list):
    total_sum = 0
    for item in nested_list:
        if isinstance(item, list):
            total_sum += sum_of_nested_list(item)  # 递归调用
        else:
            total_sum += item
    return total_sum

@app.route("/multi_step_err_infor/<data_name>/<model_names>/<step_len>")
@cross_origin()
def multi_step_err_infor (data_name, model_names, step_len):
    model_names = json.loads(model_names)
    step_len = int(step_len)
    global step_err_infor_all
    step_err_infor_all = OrderedDict()
    for model_name in model_names:
        if len(model_name) == 0: continue
        step_err_infor_all[model_name] = OrderedDict()
        residuals = all_prediction[model_name] - all_truth[model_name]
        min_limit = np.min(residuals)
        max_limit = np.max(residuals)
        
        n_sample, n_steps, n_locs = residuals.shape
        step_err_infor_list = []
        
        limit_lower_whisker = 0
        limit_upper_whisker = 0
        limit_lower_out_whisker = 0
        limit_upper_out_whisker = 0
        for i in range(n_steps):
            step_err_infor = OrderedDict()
            cur_residuals = np.ravel(residuals[:, i, :])
            # 计算箱线图的统计指标
            percentiles = np.percentile(cur_residuals, [25, 50, 75])
            # lower_whisker = np.min(cur_residuals[cur_residuals >= percentiles[0] - 1.5 * (percentiles[2] - percentiles[0])])
            # upper_whisker = np.max(cur_residuals[cur_residuals <= percentiles[2] + 1.5 * (percentiles[2] - percentiles[0])])
            lower_whisker = percentiles[0] - 1.5 * (percentiles[2] - percentiles[0])
            upper_whisker = percentiles[2] + 1.5 * (percentiles[2] - percentiles[0])
            lower_out_whisker = percentiles[0] - 3 * (percentiles[2] - percentiles[0])
            upper_out_whisker = percentiles[2] + 3 * (percentiles[2] - percentiles[0])
            # outliers = cur_residuals[(cur_residuals < lower_whisker) | (cur_residuals > upper_whisker)]
            if lower_whisker < limit_lower_whisker:
                limit_lower_whisker = lower_whisker
            if upper_whisker > limit_upper_whisker:
                limit_upper_whisker = upper_whisker
            if lower_out_whisker < limit_lower_out_whisker:
                limit_lower_out_whisker = lower_out_whisker
            if upper_out_whisker > limit_upper_out_whisker:
                limit_upper_out_whisker = upper_out_whisker
            
            step_err_infor['percentiles'] = percentiles.tolist()
            step_err_infor['lower_whisker'] = lower_whisker
            step_err_infor['upper_whisker'] = upper_whisker
            # step_err_infor['outliers'] = outliers.tolist()
            step_err_infor_list.append(step_err_infor)
        
        min_limit = -math.ceil(abs(min_limit) / step_len) * step_len
        max_limit = math.ceil(max_limit / step_len) * step_len
        limit_lower_whisker = -math.ceil(abs(limit_lower_whisker) / step_len) * step_len
        limit_upper_whisker = math.ceil(limit_upper_whisker / step_len) * step_len
        limit_lower_out_whisker = -math.ceil(abs(limit_lower_out_whisker) / step_len) * step_len
        limit_upper_out_whisker = math.ceil(limit_upper_out_whisker / step_len) * step_len
        
        mild_pos_outliers_nums = []
        mild_neg_outliers_nums = []
        extreme_pos_outliers_nums = []
        extreme_neg_outliers_nums = []
        for i in range(n_steps):
            cur_residuals = np.ravel(residuals[:, i, :])
            mild_pos_outliers = cur_residuals[(cur_residuals > upper_whisker) & (cur_residuals < limit_upper_out_whisker)]
            mild_neg_outliers = cur_residuals[(cur_residuals < limit_lower_whisker) & (cur_residuals > limit_lower_out_whisker)]
            extreme_pos_outliers = cur_residuals[cur_residuals > limit_upper_out_whisker]
            extreme_neg_outliers = cur_residuals[cur_residuals < limit_lower_out_whisker]
            step_err_infor_list[i]['mild_pos_outliers_num'] = mild_pos_outliers.size
            step_err_infor_list[i]['mild_neg_outliers_num'] = mild_neg_outliers.size
            step_err_infor_list[i]['extreme_pos_outliers_num'] = extreme_pos_outliers.size
            step_err_infor_list[i]['extreme_neg_outliers_num'] = extreme_neg_outliers.size
            mild_pos_outliers_nums.append(mild_pos_outliers.size)
            mild_neg_outliers_nums.append(mild_neg_outliers.size)
            extreme_pos_outliers_nums.append(extreme_pos_outliers.size)
            extreme_neg_outliers_nums.append(extreme_neg_outliers.size)

        val_bins = [min_limit, limit_lower_out_whisker]
        mid_bins = []
        for bin_edge in range(limit_lower_whisker, limit_upper_whisker+1, step_len):
            mid_bins.append(bin_edge)
            val_bins.append(bin_edge)
        val_bins.append(limit_upper_out_whisker)
        val_bins.append(max_limit)
        mild_outliers_num_min = min([min(mild_pos_outliers_nums), min(mild_neg_outliers_nums)])
        mild_outliers_num_max = max([max(mild_pos_outliers_nums), max(mild_neg_outliers_nums)])
        extreme_outliers_num_min = min([min(extreme_pos_outliers_nums), min(extreme_neg_outliers_nums)])
        extreme_outliers_num_max = max([max(extreme_pos_outliers_nums), max(extreme_neg_outliers_nums)])
        
        step_err_infor_all[model_name] = {
            'mid_bins': mid_bins,
            'val_bins': val_bins,
            'mild_outliers_num_range': [mild_outliers_num_min, mild_outliers_num_max],
            'extreme_outliers_num_range': [extreme_outliers_num_min, extreme_outliers_num_max],
            'step_err_infor_list': step_err_infor_list
        }
    
    return step_err_infor_all

@app.route("/phase_data/<phase_id>")
@cross_origin()
def get_phase_data (phase_id):
    phase_id = int(phase_id)
    model_name = list(phases.keys())[0]
    cur_phase = phases[model_name][phase_id]
    target_val_series = raw_dataset[:,:,:]
    
    cur_phase_data = target_val_series[cur_phase['start']:(cur_phase['end']+1)].tolist()
    
    return {'phase_raw_data': cur_phase_data}

def val_series_to_flag_series(val_series, threshold):
    bool_list = [[[bool(value > threshold) for value in sublist] for sublist in mainlist] for mainlist in val_series]
    return bool_list

def get_interval_id(value, thresholds):
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return int(i-1)  # 返回区间ID，从0开始
    return len(thresholds)-1  # 如果value大于所有阈值，返回最后一个区间ID

def val_series_to_level_series(val_series, levels):
    interval_list = [[[get_interval_id(value, levels) for value in sublist] for sublist in mainlist] for mainlist in val_series]
    return interval_list
    
@app.route("/phase_details/<model_names>/<phase_id>/<bin_edges>/<focused_val>/<focused_scopes>/<focused_scope_id>")
@cross_origin()
def get_phase_details (model_names, phase_id, bin_edges, focused_val, focused_scopes, focused_scope_id):
    model_names = json.loads(model_names)
    phase_id = int(phase_id)
    focused_val = float(focused_val)
    bin_edges = json.loads(bin_edges)
    focused_scopes = json.loads(focused_scopes)
    focused_scope_id = int(focused_scope_id)
    cur_focused_scope = []
    if focused_scope_id == 0:
        cur_focused_scope = [0, cur_run_config['output_window']]
    else:
        cur_focused_scope = focused_scopes[focused_scope_id-1]
    
    global phases
    global raw_dataset
    global truth_val_series, pred_val_series
    
    # phase_data = target_val_series[phase_infor['start']:(phase_infor['end']+1)]
    phase_details = OrderedDict()
    phase_metadata = OrderedDict()
    tmp_phase = phases[model_names[0]][phase_id]
    for attr in phase_metadata:
        phase_metadata[attr] = metadata_raw[attr][tmp_phase['start']:(tmp_phase['end']+1)].tolist()
    series_start = tmp_phase['start'] - cur_run_config['input_window']
    series_end = tmp_phase['end'] - cur_run_config['input_window']
    phase_truth_val = truth_val_series[model_names[0]][series_start:(series_end+1)]
    phase_truth_flag = val_series_to_flag_series(phase_truth_val, cur_run_config['focus_th'])
    phase_truth_level = val_series_to_level_series(phase_truth_val, cur_run_config['focus_levels'])
    time_phase_truth_flag = get_time_list(phase_truth_flag)
    time_phase_truth_level = get_time_list(phase_truth_level)
    phase_raw_data = raw_dataset[:,:,0][tmp_phase['start']:(tmp_phase['end']+1)]
    phase_raw_flag = (phase_raw_data > focused_val).astype(int)
    phase_raw_level = np.digitize(raw_dataset[:,:,0][tmp_phase['start']:(tmp_phase['end']+1)], bins=bin_edges).astype(int) - 1
    
    for j in range(len(phase_truth_val)):
        phase_truth_val[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_truth_val[j]]
        phase_truth_flag[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_truth_flag[j]]
        phase_truth_level[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_truth_level[j]]
    
    for model_name in model_names:
        if len(model_name) == 0: continue
        cur_phase = phases[model_name][phase_id]
        step_start = cur_phase['start']
        step_end = cur_phase['end']
        
        series_start = step_start - cur_run_config['input_window']
        series_end = step_end - cur_run_config['input_window']
        phase_pred_val = pred_val_series[model_name][series_start:(series_end+1)]
        phase_pred_flag = val_series_to_flag_series(phase_pred_val, cur_run_config['focus_th'])
        phase_pred_level = val_series_to_level_series(phase_pred_val, cur_run_config['focus_levels'])
        
        for j in range(len(phase_truth_val)):
            phase_pred_val[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_pred_val[j]]
            phase_pred_flag[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_pred_flag[j]]
            phase_pred_level[j] = [[item for item in sublist[cur_focused_scope[0]:cur_focused_scope[1]]] for sublist in phase_pred_level[j]]
        
        time_phase_pred_flag = get_time_list(phase_pred_flag)
        time_phase_pred_level = get_time_list(phase_pred_level)
        
        time_pod = []
        time_far = []
        time_multi_accuracy = []
        temporal_residuals_abs = []
        st_binary_label = copy.deepcopy(phase_truth_val)
        st_multi_label = copy.deepcopy(phase_truth_val)
        st_residuals = copy.deepcopy(phase_truth_val)
        
        # phase_truth_level_tmp = np.digitize(np.array(phase_truth_val), bins=bin_edges).astype(int) - 1
        # phase_truth_flag_tmp = (np.array(phase_truth_val) > 115).astype(int)
        
        for i in range(len(phase_truth_val)):
            residuals_abs_list = []
            if len(time_phase_truth_flag[i]) > 0:
                time_pod.append(float(loss.compute_POD(np.array(time_phase_pred_flag[i]), np.array(time_phase_truth_flag[i]))))
                time_far.append(float(loss.compute_FAR(np.array(time_phase_pred_flag[i]), np.array(time_phase_truth_flag[i]))))
            else:
                time_pod.append(-1)
                time_far.append(-1)
            time_multi_accuracy.append(float(loss.multi_accuracy_global(np.array(time_phase_pred_level[i]), np.array(time_phase_truth_level[i]))))
            for j in range(len(phase_truth_val[i])):
                if len(phase_pred_val[i][j]) > 0 and len(phase_truth_val[i][j]) > 0:
                    residuals_abs_list.extend(np.abs(np.array(phase_pred_val[i][j]) - np.array(phase_truth_val[i][j])).tolist())
                else:
                    residuals_abs_list.extend([-1])
                for k in range(len(phase_truth_val[i][j])):
                    if phase_pred_flag[i][j][k] == 1 and phase_truth_flag[i][j][k] == 1: 
                        st_binary_label[i][j][k] = 1  #hit
                    elif phase_pred_flag[i][j][k] == 0 and phase_truth_flag[i][j][k] == 0:
                        st_binary_label[i][j][k] = -1  #neg correct
                    elif phase_pred_flag[i][j][k] == 1 and phase_truth_flag[i][j][k] == 0:
                        st_binary_label[i][j][k] = 2  #false alarm
                    elif phase_pred_flag[i][j][k] == 0 and phase_truth_flag[i][j][k] == 1:
                        st_binary_label[i][j][k] = -2  #miss
                    if phase_pred_level[i][j][k] == phase_truth_level[i][j][k]:
                        st_multi_label[i][j][k] = 0  #估计准确 
                    elif phase_pred_level[i][j][k] < phase_truth_level[i][j][k]:
                        st_multi_label[i][j][k] = -1  #低估
                    elif phase_pred_level[i][j][k] > phase_truth_level[i][j][k]:
                        st_multi_label[i][j][k] = 1  #高估
                    st_residuals[i][j][k] = phase_pred_val[i][j][k] - phase_truth_val[i][j][k]
                
            temporal_residuals_abs.append(np.mean(np.array(residuals_abs_list)))
        temporal_residuals_abs_max = max(temporal_residuals_abs)
        
        phase_truth_space_level = []
        phase_pred_space_level = []
        # for i in range(len(phase_truth_level[0])):
        #     phase_truth_space_level.append([])
        #     phase_pred_space_level.append([])
        #     for j in range(len(phase_truth_level)):
        #         phase_truth_space_level[i].extend(phase_truth_level[j][i])
        #         phase_pred_space_level[i].extend(phase_pred_level[j][i])
        
        # space_level_confusion = np.zeros((len(bin_edges)-1, len(bin_edges)-1, len(phase_truth_level[0])))
        global_level_confusion = np.zeros((len(bin_edges)-1, len(bin_edges)-1))
        # for i in range(len(phase_truth_space_level)):
        #     for j in range(len(phase_truth_space_level[i])):
        #         x_id = phase_truth_space_level[i][j]
        #         y_id = phase_pred_space_level[i][j]
        #         space_level_confusion[x_id][y_id][i] += 1
        #         global_level_confusion[x_id][y_id] += 1
        
        # 对事件属性分bin
        event_attr_bin_edges = compute_phases_attr_metric_bins(cur_phase['phase_event_objs'])
        print('event_attr_bin_edges', event_attr_bin_edges)
        
        phase_details[model_name] = {
            'sel_phase_id': int(phase_id),
            # 'space_level_confusion': space_level_confusion.tolist(),
            # 'global_level_confusion': global_level_confusion.tolist(),
            # 'space_confusion_max_cnt': int(np.max(space_level_confusion)),
            'temporal_residuals_abs_max': temporal_residuals_abs_max,
            'temporal_residuals_abs': temporal_residuals_abs,
            'time_pod': time_pod,
            'time_far': time_far,
            'time_multi_accuracy': time_multi_accuracy,
            # 'phase_truth_val': phase_truth_val,
            'phase_pred_val': phase_pred_val,
            'phase_pred_flag': phase_pred_flag,
            'phase_pred_level': phase_pred_level,
            'st_binary_label': st_binary_label,
            'st_multi_label': st_multi_label,
            'st_residuals': st_residuals,
            'event_attr_bin_edges': event_attr_bin_edges
        }
    # print(phase_details)
    
    return {
        'phase_details': phase_details,
        'phase_metadata': phase_metadata,
        'phase_raw_level': phase_raw_level.tolist(),
        'phase_raw_val': phase_raw_data.tolist(),
        'phase_raw_flag': phase_raw_flag.tolist(),
        }

@app.route("/loc_instance_infor/<model_names>/<phase_id>/<step_id>/<loc_id>")
@cross_origin()
def get_loc_instance_infor (model_names, phase_id, step_id, loc_id):
    model_names = json.loads(model_names)
    phase_id = int(phase_id)
    step_id = int(step_id)
    loc_id = int(loc_id)
    
    phase_start = phases[model_names[0]][phase_id]['start']
    preds_with_prev = {}
    data_step_id = phase_start+step_id
    prev_truth = [step_data[loc_id] for step_data in raw_dataset[data_step_id-12:(data_step_id+1)].tolist()]
    
    for model_name in model_names:
        if len(model_name) == 0: continue
        preds_with_prev[model_name] = []
        for i in range(cur_run_config['output_window']):
            ins_id = phase_start + step_id - cur_run_config['input_window']
            pred_seq = [step_vals[loc_id] for step_vals in all_prediction[model_name][ins_id-i][:(i+1)].tolist()]
            preds_with_prev[model_name].append(pred_seq)
    
    return {
        'prev_truth': prev_truth,
        'preds_with_prev': preds_with_prev
        }

@app.route("/save_subgroup_collections/<data_name>/<model_name>/<subgroup_collections>/<cur_configs>")
@cross_origin()
def save_subgroup_collections (data_name, model_name, subgroup_collections, cur_configs):
    subgroup_collections = json.loads(subgroup_collections)
    cur_configs = json.loads(cur_configs)
    
    with open('./saved_subgroup_collections.json', 'r', encoding='utf-8') as f:
        saves_collections = json.load(f)
    
    if data_name not in saves_collections:
        saves_collections[data_name] = {}
    
    slice_indices_file = './point_slices/slice_indices.json'
    with open(slice_indices_file, 'r', encoding='utf-8') as f:
        slice_indices = json.load(f)
    cached_slices_configs = slice_indices[data_name][model_name]
    cur_slice_config = {
        "focus_th": cur_run_config['focus_th'],
        "range_params": cur_run_config['range_params'],
        "slice_params": cur_run_config['slice_params']
    }
    cached_slice_id = None
    for slice_id in cached_slices_configs:
        if cur_slice_config == cached_slices_configs[slice_id]:
            cached_slice_id = slice_id
            break
    saves_collections[data_name][f'{model_name}_{cached_slice_id}'] = subgroup_collections
    
    with open('./saved_subgroup_collections.json', 'w', encoding='utf-8') as f:
        json.dump(saves_collections, f)
    
    return 'save ok'

@app.route("/save_phase_collections/<data_name>/<phase_collections>/<cur_configs>")
@cross_origin()
def save_phase_collections (data_name, phase_collections, cur_configs):
    phase_collections = json.loads(phase_collections)
    cur_configs = json.loads(cur_configs)
    
    with open('./saved_phase_collections.json', 'r', encoding='utf-8') as f:
        saves_collections = json.load(f)
    
    if data_name not in saves_collections:
        saves_collections[data_name] = {}
    
    focus_th = cur_run_config['focus_th']
    min_length = cur_run_config['phase_params']['min_len']
    max_gap_len = cur_run_config['phase_params']['max_gap_len']
    params_str = generate_file_label(cur_run_config['phase_params']['attributes'])
    saves_collections[data_name][f'{focus_th}_{min_length}_{max_gap_len}_{params_str}'] = subgroup_collections
    
    with open('./saved_subgroup_collections.json', 'w', encoding='utf-8') as f:
        json.dump(saves_collections, f)
    
    return 'save ok'

def dynamic_rounding(values):
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val > 10:
        decimal_places = 1  # 如果范围较大，可以舍入到整数
    elif range_val > 1:
        decimal_places = 2  # 中等范围，保留1位小数
    elif range_val > 0.1:
        decimal_places = 3  # 范围较小，保留2位小数
    else:
        decimal_places = 4  # 范围非常小，保留3位小数

    rounded_values = [round(v, decimal_places) for v in values]
    return rounded_values

def compute_phases_attr_metric_bins(events):
    attr_metric_bin_edges = {}
    for attr in cur_run_config['event_params']['attributes']:
        if isinstance(events[0][attr], numbers.Number):
            attr_metric_bin_edges[attr] = []
            attr_vals = [event[attr] for event in events]
            attr_data = np.reshape(np.array(attr_vals), (-1, 1))
            sup_num = math.floor(len(attr_vals) * 0.05)
            tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=1)
            # tree = DecisionTreeRegressor(max_leaf_nodes=10)
            tree.fit(attr_data, attr_data)
            bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
            bin_edges.sort()
            data_min = math.floor(np.min(attr_data))
            data_max = math.ceil(np.max(attr_data))
            all_bin_edges = [data_min]
            all_bin_edges.extend(bin_edges)
            all_bin_edges.append(data_max)
            print("all_bin_edges: ", all_bin_edges)
            attr_metric_bin_edges[attr] = dynamic_rounding(all_bin_edges)
    return attr_metric_bin_edges

@app.route("/instance_seqs/<model_names>/<stamp_id>/<loc_id>/<focused_scope>")
@cross_origin()
def get_instance_seqs (model_names, stamp_id, loc_id, focused_scope):
    global truth_val_series, pred_val_series
    global all_truth, all_prediction, input_dataset_infor
    global raw_dataset
    focused_scope = json.loads(focused_scope)
    focused_scope[0] -= 1
    # focused_scope[1] -= 1
    model_names = json.loads(model_names)
    stamp_id = int(stamp_id)
    loc_id = int(loc_id)
    month_pre = math.ceil(stamp_id / (24*31))
    instance_step_id = int(stamp_id - cur_run_config["input_window"]*month_pre - (cur_run_config["output_window"]-1)*(month_pre-1))
    # instance_step_id = int(stamp_id - cur_run_config["input_window"]*month_pre)
    
    input_dataset = input_dataset_infor['items'][:,:,:,0]
    err_step_id = stamp_id - month_pre * cur_run_config["input_window"]
    
    instance_seqs_objs = OrderedDict()
    for model_name in model_names:
        if len(model_name) == 0: continue
        instance_seqs_objs[model_name] = {}
        seq_num = len(truth_val_series[model_name][err_step_id][0])
        instance_seqs = []
        instance_seqs_space = []
        pred_seqs_space = []
        truth_seq = []
        truth_seq_space = []
        pre_interval = cur_run_config["output_window"] - seq_num
        for i in range(seq_num):
            pred_step_num = i + pre_interval
            truth_step_num = cur_run_config['input_window'] - pred_step_num
            instance_seq = []
            instance_seq_space = []
            pred_seq_space = []
            instance_id = instance_step_id - pred_step_num
            # for j in range(cur_run_config["input_window"]):
            for j in range(focused_scope):
                step_id = j
                if j < truth_step_num:
                    step_data = float(input_dataset[instance_id][j+i][loc_id])
                    step_data_space = input_dataset[instance_id][j+i].tolist()
                else:
                    step_data = float(all_prediction[model_name][instance_id][j-truth_step_num][loc_id])
                    step_data_space = all_prediction[model_name][instance_id][j-truth_step_num].tolist()
                    pred_seq_space.append(step_data_space)
                instance_seq.append(step_data)
                instance_seq_space.append(step_data_space)
            instance_seqs.append(instance_seq)
            instance_seqs_space.append(instance_seq_space)
            pred_seqs_space.append(pred_seq_space)
        # print(raw_dataset.shape)
        truth_seq = raw_dataset[stamp_id-cur_run_config['input_window']:stamp_id,loc_id,0].tolist()
        truth_seq_space = raw_dataset[stamp_id-cur_run_config['input_window']:stamp_id,:,0].tolist()
        output_pred_steps = pred_val_series[model_name][err_step_id][loc_id]
        output_pred_steps_space = pred_val_series[model_name][err_step_id]
        pred_seqs_space.append(pred_val_series[model_name][err_step_id])
        output_truth_steps = truth_val_series[model_name][err_step_id][loc_id]
        output_truth_steps_space = truth_val_series[model_name][err_step_id]
        
        val_min = 0
        seq_max = np.max(np.array(instance_seqs_space))
        output_max = np.max(np.array(output_truth_steps_space))
        val_max = max(seq_max, output_max)
        
        instance_seqs_objs[model_name] = {
            "stamp_id": stamp_id,
            "truth_seq": truth_seq,
            "truth_seq_space": truth_seq_space,
            "instance_seqs": instance_seqs,
            "instance_seqs_space": instance_seqs_space,
            "output_pred_steps": output_pred_steps,
            "output_pred_steps_space": output_pred_steps_space,
            "output_truth_steps": output_truth_steps,
            "output_truth_steps_space": output_truth_steps_space,
            "pred_seqs_space": pred_seqs_space,
            "val_range": [val_min, val_max]
        }
    
    return instance_seqs_objs

@app.route("/loc_features/<loc_id>/<stamp_id>")
@cross_origin()
def get_loc_features (loc_id, stamp_id):
    loc_id = int(loc_id)
    stamp_id = int(stamp_id)
    
    global cur_dataset_infor
    global timeStrs
    all_features = cur_dataset_infor['Features']
    show_features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "U", "V", "TEMP", "RH", "PSFC"]
    # 获取展示feature的ids
    feature_seqs = {}
    for feature in show_features:
        feature_id = all_features.index(feature)
        feature_seqs[feature] = []
        for i in range(stamp_id-cur_run_config['input_window'], stamp_id+1):
            feature_obj = {
                'stamp': timeStrs[i],
                'val': raw_dataset[i][loc_id][feature_id]
            }
            feature_seqs[feature].append(feature_obj)
    
    return jsonify(feature_seqs)


@app.route("/save_error_config/<cur_task>/<config_name>/<config_description>/<failure_rules>/<scope_seg_th>")
@cross_origin()
def save_new_error_config(cur_task, config_name, config_description, failure_rules, scope_seg_th):
    failure_rules = json.loads(failure_rules)
    scope_seg_th = int(scope_seg_th)
    error_configs = None
    with open('./error_configs.json', 'r', encoding='utf-8') as f:
        error_configs = json.load(f)
    error_configs[cur_task][config_name] = {}
    error_configs[cur_task][config_name]['failure_rules'] = failure_rules
    error_configs[cur_task][config_name]['time_scope_steps'] = scope_seg_th
    error_configs[cur_task][config_name]['description'] = config_description
    with open('./error_configs.json', 'w', encoding='utf-8') as f:
        json.dump(error_configs, f)
    
    return 'Config has been saved successfully!'

if __name__ == '__main__':
    # from gevent.pywsgi import WSGIServer
    # http_server = WSGIServer(('0.0.0.0', 1024), app)
    # http_server.serve_forever()
    app.run(host='0.0.0.0', port=1024)
