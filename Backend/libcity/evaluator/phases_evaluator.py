import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
import numpy as np
import math
import random
import copy
import numbers
from sklearn.tree import DecisionTreeRegressor
import hashlib

class PhaseEvaluator():
    def __init__(self, raw_data, model_name, config, phases, truth_val_series, pred_val_series, residuals_series, residual_bins, forecast_scopes):
        self.config = config
        self.raw_data = raw_data
        self.model_name = model_name
        self.phases = phases
        self.dataset = self.config.get('dataset', '')
        
        self.truth_val_series = truth_val_series
        self.pred_val_series = pred_val_series
        self.residuals_series = residuals_series
        
        self.residual_bins = residual_bins
        
        self.forecast_scopes = forecast_scopes
    
    def flatten_list(self, lst):
        flat_list = []
        for sublist in lst:
            if isinstance(sublist, list):
                flat_list.extend(self.flatten_list(sublist))
            else:
                flat_list.append(sublist)
        return flat_list
    
    def get_space_list(self, lst):
        space_list = []
        for i in range(len(lst[0])):
            space_list.append([])
            for j in range(len(lst)):
                space_list[i].extend(lst[j][i])
        return space_list
    
    def space_res_abs(self, phase_truth_series, phase_pred_series):
        space_phase_truth_val = np.array(self.get_space_list(phase_truth_series))
        space_phase_pred_val = np.array(self.get_space_list(phase_pred_series))
        return loss.space_abs_residual(space_phase_pred_val, space_phase_truth_val).tolist()
    
    def res_distribution(self, phase_truth_series, phase_pred_series):
        flat_phase_truth_series = np.array(self.flatten_list(phase_truth_series))
        flat_phase_pred_series = np.array(self.flatten_list(phase_pred_series))
        flat_phase_res_series = flat_phase_pred_series - flat_phase_truth_series
        whole_residual_bins = self.residual_bins['residual_bins']
        residual_hist, _  = np.histogram(flat_phase_res_series, bins=whole_residual_bins, density=False)
        
        # 计算mid_bins的分布
        mid_bins = self.residual_bins['mid_bins']
        mid_residuals = flat_phase_res_series[(flat_phase_res_series < whole_residual_bins[-2])&(flat_phase_res_series > whole_residual_bins[1])]
        mid_hist, _  = np.histogram(mid_residuals, bins=mid_bins, density=False)
        
        # 计算pos_extreme_bins的分布
        pos_extreme_bins = self.residual_bins['pos_extreme_bins']
        extreme_pos_outliers = flat_phase_res_series[flat_phase_res_series > whole_residual_bins[-2]]
        pos_extreme_hist, _  = np.histogram(extreme_pos_outliers, bins=pos_extreme_bins, density=False)
        
        # 计算neg_extreme_bins的分布
        neg_extreme_bins = self.residual_bins['neg_extreme_bins']
        extreme_neg_outliers = flat_phase_res_series[flat_phase_res_series < whole_residual_bins[1]]
        neg_extreme_hist, _  = np.histogram(extreme_neg_outliers, bins=neg_extreme_bins, density=False)
                
        return {'residual_hist': residual_hist.tolist(), 
            'mid_hist': mid_hist.tolist(),
            'pos_extreme_hist': pos_extreme_hist.tolist(),
            'neg_extreme_hist': neg_extreme_hist.tolist()}
        
    def phase_RMSE(self, phase_truth_series, phase_pred_series):
        flat_phase_truth_series = np.array(self.flatten_list(phase_truth_series))
        flat_phase_pred_series = np.array(self.flatten_list(phase_pred_series))
        return round(float(loss.rmse_np(flat_phase_truth_series, flat_phase_pred_series)), 6)
    
    def phase_ACC(self, phase_truth_series, phase_pred_series):
        phase_truth_space_val = []
        phase_pred_space_val = []
        for j in range(len(phase_truth_series[0])):
            phase_truth_space_val.append([])
            phase_pred_space_val.append([])
            for k in range(len(phase_truth_series)):
                phase_truth_space_val[j].extend(phase_truth_series[k][j])
                phase_pred_space_val[j].extend(phase_pred_series[k][j])
        climatology = np.mean(self.raw_data[:,:,0], axis=0)
        return round(float(loss.acc_score_np(np.array(phase_pred_space_val), np.array(phase_truth_space_val), climatology[:, np.newaxis])), 6)
    
    def phase_POD(self, phase_truth_series, phase_pred_series):
        truth_flat_array = np.array(self.flatten_list(phase_truth_series))
        pred_flat_array = np.array(self.flatten_list(phase_pred_series))
        truth_flat_flag = (truth_flat_array >= self.config['focus_th']).astype(int)
        pred_flat_flag = (pred_flat_array >= self.config['focus_th']).astype(int)
        return round(float(loss.compute_POD(pred_flat_flag, truth_flat_flag)), 6)
    
    def phase_FAR(self, phase_truth_series, phase_pred_series):
        truth_flat_array = np.array(self.flatten_list(phase_truth_series))
        pred_flat_array = np.array(self.flatten_list(phase_pred_series))
        truth_flat_flag = (truth_flat_array >= self.config['focus_th']).astype(int)
        pred_flat_flag = (pred_flat_array >= self.config['focus_th']).astype(int)
        return round(float(loss.compute_FAR(pred_flat_flag, truth_flat_flag)), 6)
    
    def focus_level_accuracy(self, phase_truth_series, phase_pred_series):
        truth_flat_array = np.array(self.flatten_list(phase_truth_series))
        pred_flat_array = np.array(self.flatten_list(phase_pred_series))
        truth_flat_level = np.digitize(truth_flat_array, bins=self.config['focus_levels']).astype(int) - 1
        pred_flat_level = np.digitize(pred_flat_array, bins=self.config['focus_levels']).astype(int) - 1
        _, multi_accuracy_list = loss.multi_accuracy_individual(pred_flat_level, truth_flat_level, len(self.config['focus_levels'])-1)
        return multi_accuracy_list
    
    def call_method_by_name(self, method_name, phase_truth_series, phase_pred_series, *args, **kwargs):
        attr_val = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            attr_val = method(phase_truth_series, phase_pred_series, *args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return attr_val
    
    def compute_phases_err_metrics(self):
        for i, phase in enumerate(self.phases):
            step_start = phase['start']
            step_end = phase['end']
            series_start = phase['start'] - self.config['input_window']
            series_end = phase['end'] - self.config['input_window']
            phase_truth_series = self.truth_val_series[series_start:(series_end+1)]
            phase_pred_series = self.pred_val_series[series_start:(series_end+1)]
            
            for metric in self.config['phase_params']['error_metrics']:
                phase[metric] = self.call_method_by_name(metric, phase_truth_series, phase_pred_series)
                
        return 'ok'
    
    def dynamic_rounding(self, values):
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
    
    def compute_phases_attr_metric_bins(self):
        attr_metric_bin_edges = {}
        for attr in self.config['phase_params']['attributes']:
            if isinstance(self.phases[0][attr], numbers.Number):
                attr_metric_bin_edges[attr] = []
                attr_vals = [phase[attr] for phase in self.phases]
                attr_data = np.reshape(np.array(attr_vals), (-1, 1))
                sup_num = math.floor(len(attr_vals) * 0.05)
                tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=sup_num)
                # tree = DecisionTreeRegressor(max_leaf_nodes=10)
                tree.fit(attr_data, attr_data)
                bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
                bin_edges.sort()
                data_min = math.floor(np.min(attr_data))
                data_max = math.ceil(np.max(attr_data))
                all_bin_edges = [data_min]
                all_bin_edges.extend(bin_edges)
                all_bin_edges.append(data_max)
                attr_metric_bin_edges[attr] = self.dynamic_rounding(all_bin_edges)
        for attr in self.config['phase_params']['error_metrics']:
            # if isinstance(self.phases[0][attr], numbers.Number):
            if attr == 'phase_RMSE':
                attr_metric_bin_edges[attr] = []
                attr_vals = [phase[attr] for phase in self.phases]
                attr_data = np.reshape(np.array(attr_vals), (-1, 1))
                sup_num = math.floor(len(attr_vals) * 0.05)
                tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=sup_num)
                # tree = DecisionTreeRegressor(max_leaf_nodes=10)
                tree.fit(attr_data, attr_data)
                bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
                bin_edges.sort()
                data_min = math.floor(np.min(attr_data))
                data_max = math.ceil(np.max(attr_data))
                all_bin_edges = [data_min]
                all_bin_edges.extend(bin_edges)
                all_bin_edges.append(data_max)
                attr_metric_bin_edges[attr] = self.dynamic_rounding(all_bin_edges)
        
        def generate_file_label(attributes):
            sorted_attributes = sorted(attributes)
            attributes_str = "_".join(sorted_attributes)
            file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
            file_label = f"{file_hash[:8]}"
            return file_label
        params_label = generate_file_label(self.config['phase_params']['attributes'])
        bin_edges_file = f"./phases_data/{self.dataset}_{self.model_name}_{float(self.config['focus_th'])}_{self.config['phase_params']['min_len']}_{self.config['phase_params']['max_gap_len']}_{params_label}.json"
        
        with open(bin_edges_file, 'w', encoding='utf-8') as f:
                json.dump(attr_metric_bin_edges, f)
        
        return attr_metric_bin_edges
        
        
        # 计算哪些indicators：
        # 不同forecast scopes下的二分指标、level指标、residual/val指标
        
        # 事件上的指标？
        # 全集 -> 阶段 -> 时间步 -> 实体（事件）-> 点：各个层级之间不应该是孤立的？？？
        # 各层级关联起来，有什么好处？？探索性分析？
        