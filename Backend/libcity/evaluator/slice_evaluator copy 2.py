import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from analysis.seg_range_point import seg_range_point
from analysis.extract_subset_iterative_point import slicing_with_polarity_pruning
from analysis.extract_subset_iterative_point import slicing_with_err_th
from scipy.stats import norm
from scipy.stats import kstest

class SliceEvaluator():
    def __init__(self, config, data_scope, metadata_series, residuals_series, residual_bins, phases, forecast_scopes):
        self.config = config
        self.data_scope = data_scope
        self.metadata_series = metadata_series
        self.residuals_series = residuals_series
        self.residual_bins = residual_bins
        self.phases = phases
        self.forecast_scopes = forecast_scopes
        
        self.range_params = config.get('range_params', {})
        self.slice_params = config.get('slice_params', {})
        self.scope_residuals = []
        self.scope_metadata = {}
        self.metadata_ranges = {}
        self.ranges_infor = {}
        self.slices_infor = {}
        self.ranges_indices = {}
        self.slices_indices = {}
        # self.global_st_indices_list = []
        # self.scope_st_indices_list = []
    
    def flatten_list(self, multi_list):
        return np.array([item for sublist1 in multi_list for sublist2 in sublist1 for item in sublist2])
    
    def process_global_st_indices(self):
        for i in range(len(self.residuals_series)):
            self.global_st_indices_list.append([])
            for j in range(len(self.residuals_series[i])):
                self.global_st_indices_list[i].append([])
                for k in range(len(self.residuals_series[i][j])):
                    self.global_st_indices_list[i][j].append([i,j,k])
    
    def process_scope_data(self):
        if self.data_scope == 'All_Data':
            self.scope_residuals = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.residuals_series]))
            for attr in self.config['point_metadata']:
                self.scope_metadata[attr] = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.metadata_series[attr]]))
            # self.scope_st_indices_list = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.global_st_indices_list]))
        elif self.data_scope == 'All_Phases':
            phase_ins_ids = []
            for phase_id, phase in enumerate(self.phases):
                step_start = phase['start'] - self.config['input_window']
                step_end = phase['end'] - self.config['input_window']
                phase_ins_ids += list(range(step_start, step_end+1))
            self.scope_residuals = [self.residuals_series[ins_id] for ins_id in phase_ins_ids]
            self.scope_residuals = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.scope_residuals]))
            for attr in self.config['point_metadata']:
                self.scope_metadata[attr] = [self.metadata_series[attr][ins_id] for ins_id in phase_ins_ids]
                self.scope_metadata[attr] = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.scope_metadata[attr]]))
            # phase_st_indices = [self.global_st_indices_list[ins_id] for ins_id in phase_ins_ids]
            # self.scope_st_indices_list = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in phase_st_indices]))
        elif self.data_scope == 'Selected_Phase':
            phase_id = self.slice_params['phase_id']
            phase = self.phases[phase_id]
            step_start = phase['start'] - self.config['input_window']
            step_end = phase['end'] - self.config['input_window']
            self.scope_residuals = self.residuals_series[step_start:step_end]
            self.scope_residuals = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.scope_residuals]))
            for attr in self.config['point_metadata']:
                self.scope_metadata[attr] = self.metadata_series[attr][step_start:step_end]
                self.scope_metadata[attr] = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in self.scope_metadata[attr]]))
            # phase_st_indices = self.global_st_indices_list[step_start:step_end]
            # self.scope_st_indices_list = np.array(self.flatten_list([[loc_list for loc_list in step_list] for step_list in phase_st_indices]))
    
    def divide_range_by_step(self, val_range, step, bin_num):
        start = val_range[0]
        end = val_range[1]
        # 将起始值和结束值转换为以step为单位的整数  
        start_int = math.floor(start / step)
        end_int = math.ceil(end / step)
        # 计算整个范围的长度（以step为单位）  
        range_length_int = end_int - start_int
        # 如果范围长度为0，则直接返回包含起始值和结束值的列表  
        if range_length_int == 0:
            return [start_int, end_int]
        # 计算每一份的整数长度（以step为单位）
        part_length_int = math.floor(range_length_int / (bin_num))
        # 如果不是均匀分割，确保最后一份至少有一个单位长度
        # if range_length_int % (bin_num) != 0:  
        #     part_length_int -= 1  
        # 初始化分界值列表，包含起始值  
        boundaries = [math.floor(start)]  
        # 当前整数位置  
        current_int = start_int + 1  
        # 循环计算剩余的分界值（不包括结束值）  
        for _ in range(1, bin_num):  # 只需要计算到第9个分界值，因为第10个是结束值  
            # 计算下一个分界值的整数位置  
            current_int += part_length_int  
            # 将整数位置转换回浮点数，并确保是step的整数倍  
            current_value = step * current_int  
            # 添加分界值到列表  
            boundaries.append(current_value)  
        # 添加结束值到列表  
        boundaries.append(math.ceil(end))  
        return boundaries
    
    def is_heavy_tailed(self, data):
        """Check if data is heavy-tailed by comparing it to a normal distribution."""
        statistic, p_value = kstest(data, 'norm')
        print('statistic, p_value', statistic, p_value)
        return statistic > 0.05
    
    def find_split_point(self, data, sup_num):
        """Find the optimal split point that divides data into two parts."""
        split_points = []
        
        def partition_recursive(data):
            mean = round(np.mean(data))
            left_part = data[data <= mean]
            right_part = data[data > mean]
            while left_part.size < sup_num:
                mean += 1
                left_part = data[data <= mean]
            while right_part.size < sup_num:
                mean -= 1
                right_part = data[data > mean]
            split_points.append(mean)
            
            if right_part.size >= 2*sup_num and self.is_heavy_tailed(right_part):
                partition_recursive(right_part)
            else: return

        partition_recursive(data)

        return split_points
    
    def range_division(self):
        err_th = self.range_params['err_th']
        ranges_infor_file = f"./point_slices/ranges_{self.config['dataset']}_{self.range_params['mode']}_{str(self.range_params['bin_num'])}.json"
        if os.path.exists(ranges_infor_file):
            with open(ranges_infor_file, 'r', encoding='utf-8') as f:
                self.metadata_ranges = json.load(f)
        else:
            for attr in self.config['point_metadata']:
                self.metadata_ranges[attr] = {'range': [np.min(self.scope_metadata[attr]), np.max(self.scope_metadata[attr])]}
                
            
            if self.range_params['mode'] == 'equal_val':
                bin_num =  int(self.range_params['bin_num'])
                for attr in self.scope_metadata.keys():
                    bin_edges = self.divide_range_by_step(self.metadata_ranges[attr]['range'], 1, bin_num)
                    self.metadata_ranges[attr]['bin_edges'] = bin_edges
                    self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                    # self.ranges_infor[attr] = self.metadata_ranges[attr]['bins']
            elif self.range_params['mode'] == 'equal_fre':
                bin_num =  int(self.range_params['bin_num'])
                for attr in self.scope_metadata.keys():
                    # 获取attr的数据
                    attr_data = self.scope_metadata[attr]
                    # 计算bin_edges
                    est=KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
                    est.fit(np.array(attr_data).reshape(-1, 1))
                    bin_edges=est.bin_edges_[0].tolist()
                    for i in range(len(bin_edges)):
                        if i == 0: bin_edges[i] = math.floor(bin_edges[i])
                        elif i == len(bin_edges)-1: bin_edges[i] = math.ceil(bin_edges[i])
                        else: bin_edges[i] = round(bin_edges[i])
                    self.metadata_ranges[attr]['bin_edges'] = bin_edges
                    self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                    # self.ranges_infor[attr] = self.metadata_ranges[attr]['bins']
            elif self.range_params['mode'] == 'adaptive':
                print('run adaptive range division.')
                bin_num =  int(self.range_params['bin_num'])
                def get_node_ranges(tree_, node_id, lower_bound, upper_bound):
                    children_left = tree_.children_left
                    children_right = tree_.children_right
                    threshold = tree_.threshold
                    if children_left[node_id] == children_right[node_id]:  # 叶节点
                        return [(lower_bound, upper_bound)]
                    left_ranges = get_node_ranges(tree_, children_left[node_id], lower_bound, min(upper_bound, threshold[node_id]))
                    right_ranges = get_node_ranges(tree_, children_right[node_id], max(lower_bound, threshold[node_id]), upper_bound)
                    return [(lower_bound, upper_bound)] + left_ranges + right_ranges
                
                for attr in self.scope_metadata.keys():
                    # 获取attr的数据
                    attr_data = np.reshape(self.scope_metadata[attr], (-1, 1))
                    sup_num = math.floor(self.range_params['sup_th'] * attr_data.size)
                    tree = DecisionTreeRegressor(max_leaf_nodes=bin_num, min_samples_leaf=sup_num)
                    tree.fit(attr_data, attr_data)
                    bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
                    bin_edges.sort()
                    print('bin_edges after sort:', bin_edges)
                    data_min = math.floor(np.min(attr_data))
                    data_max = math.ceil(np.max(attr_data))
                    all_bin_edges = [data_min]
                    all_bin_edges.extend(bin_edges)
                    all_bin_edges.append(data_max)
                    print('bin_edges with min max:', all_bin_edges)
                    bin_edges = [round(bin_edge) for bin_edge in all_bin_edges]
                    print('bin_edges after round:', bin_edges)
                    self.metadata_ranges[attr]['bin_edges'] = bin_edges
                    print('bin_edges', bin_edges)
                    if self.range_params['scope'] == 'Hierarchy':
                        self.metadata_ranges[attr]['bins'] = get_node_ranges(tree.tree_, 0, data_min, data_max)[1:]
                    else: 
                        self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                    # self.ranges_infor[attr] = self.metadata_ranges[attr]['bins']
                ranges_infor_file = f"./point_slices/ranges_{self.config['dataset']}_{self.range_params['mode']}_{self.range_params['scope']}_{str(self.range_params['bin_num'])}_{self.range_params['sup_th']}.json"
                if not os.path.exists(ranges_infor_file):
                    with open(ranges_infor_file, 'w', encoding='utf-8') as f:
                        json.dump(self.metadata_ranges, f)
            elif self.range_params['mode'] == 'Head-Tail-Breaks':
                print('run supervised Head/tail_Breaks division.')
                for attr in self.scope_metadata.keys():
                    sup_num = math.floor(self.range_params['sup_th'] * self.scope_metadata[attr].size)
                    bin_edges = self.find_split_point(self.scope_metadata[attr], sup_num)
                    print(attr, bin_edges)
                    self.metadata_ranges[attr]['bin_edges'] = bin_edges
                    self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                ranges_infor_file = f"./point_slices/ranges_{self.config['dataset']}_{self.range_params['mode']}_{self.range_params['sup_th']}.json"
                if not os.path.exists(ranges_infor_file):
                    with open(ranges_infor_file, 'w', encoding='utf-8') as f:
                        json.dump(self.metadata_ranges, f)
            elif self.range_params['mode'] == 'supervised':
                print('run supervised range division.')
                val_step = self.range_params['step_len']
                sup_th = self.range_params['sup_th']
                divide_level_num =  int(self.range_params['divide_level_num'])
                divide_type = "bin_num"
                ranges_infor = seg_range_point(self.metadata_ranges, self.scope_metadata, self.scope_residuals, val_step, sup_th, err_th, divide_level_num, divide_type)
                for attr in self.config['point_metadata']:
                    all_bin_edges = ranges_infor[attr]['seg_points']
                    bin_edges = [round(bin_edge/val_step[attr]) * val_step[attr] for bin_edge in all_bin_edges]
                    self.metadata_ranges[attr]['bin_edges'] = bin_edges
                    if self.range_params['scope'] == 'Hierarchy':
                        self.metadata_ranges[attr]['bins'] = ranges_infor[attr]['preorder_tree_range']
                    else:
                        self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                    print('attr', bin_edges)
                    # important_leaf_nodes = ranges_infor[attr]['important_leaf_nodes']
                    # important_leaf_nodes_range = [node['range'] for node in important_leaf_nodes]
                    # self.metadata_ranges[attr]['important_bins'] = important_leaf_nodes_range
                    # self.ranges_infor[attr] = important_leaf_nodes
                ranges_infor_file = f"./point_slices/ranges_{self.config['dataset']}_{self.range_params['mode']}_{self.range_params['scope']}_{str(sup_th)}_{str(err_th)}.json"
                if not os.path.exists(ranges_infor_file):
                    with open(ranges_infor_file, 'w', encoding='utf-8') as f:
                        json.dump(self.metadata_ranges, f)
        
        # 计算range相关的属性
        for attr in self.config['point_metadata']:
            self.ranges_infor[attr] = []
            # metadata_flatten = np.array(self.flatten_list(self.metadata_series[attr]))
            # residuals_flatten = np.array(self.flatten_list(self.residuals_series))
            metadata_flatten = self.scope_metadata[attr]
            residuals_flatten = self.scope_residuals
            residuals_flatten = np.array(self.flatten_list(self.residuals_series))
            for cur_range in self.metadata_ranges[attr]['bins']:
                range_flags = (metadata_flatten >= cur_range[0]) & (metadata_flatten <= cur_range[1])
                cur_range_infor = {}
                cur_range_infor['range'] = cur_range
                cur_range_infor['size'] = np.count_nonzero(range_flags)
                cur_range_infor['cover'] = cur_range_infor['size'] / metadata_flatten.size
                cur_range_infor['abs_residual'] = np.mean(np.abs(residuals_flatten[range_flags]))
                cur_range_infor['div'] = cur_range_infor['abs_residual'] - np.mean(np.abs(residuals_flatten))
                cur_range_infor['range_str'] = attr + '=' + str(cur_range)
                self.ranges_indices[cur_range_infor['range_str']] = range_flags
                self.ranges_infor[attr].append(cur_range_infor)
        
        return 'ok'
    
    def subgroup_mining(self):
        # self.process_global_st_indices()
        # print("finish process indices")
        self.process_scope_data()
        print("finish process scope_data")
        self.range_division()
        print("finish range division")
        err_diff_val_th = float(self.slice_params['err_diff_th'])
        subset_div_val_th = float(self.slice_params['err_th'])
        print("prepare to run function")
        print('self.ranges_infor', self.ranges_infor)
        if self.slice_params["mode"] == "Polarity_Pruning":
            scope_subsets_iterative = slicing_with_polarity_pruning(self.ranges_infor, self.scope_residuals, self.scope_metadata, self.metadata_ranges, self.forecast_scopes, self.residual_bins, self.slice_params['frequent_sup_th'], err_diff_val_th, subset_div_val_th, self.slice_params['redundancy_th'], self.slice_params['polarity'])
        elif self.slice_params["mode"] == "Error_TH":
            scope_subsets_iterative = slicing_with_err_th(self.ranges_infor, self.scope_residuals, self.scope_metadata, self.metadata_ranges, self.forecast_scopes, self.residual_bins, self.slice_params['frequent_sup_th'], err_diff_val_th, subset_div_val_th, self.slice_params['redundancy_th'])
        print('finish computing slices...')
    
        for i, subset in enumerate(scope_subsets_iterative):
            subset['subset_id'] = str(i+1)
            self.slices_indices[subset['subset_id']] = subset.pop('indices')
            if 'contain_subsets' in subset:
                for j, individual_subset in enumerate(subset['contain_subsets']):
                    individual_subset['subset_id'] = str(i+1) + '.' + str(j+1)
                    self.slices_indices[individual_subset['subset_id']] = individual_subset.pop('indices')
        self.slices_infor = {
            'data_scope': self.data_scope,
            'slices': scope_subsets_iterative
            }
        
        return 'ok'
    
    def save_ranges_slices(self):
        model_name = self.config['model'] + '-' + str(self.config['exp_id'])
        exp_id = int(random.SystemRandom().random() * 100000)
        # 读取，并构建存储参数信息的json文件
        slice_indices_file = './point_slices/slice_indices.json'
        with open(slice_indices_file, 'r') as file:
            slice_indices = json.load(file)
        if self.config['dataset'] not in slice_indices:
            slice_indices[self.config['dataset']] = {}
        if model_name not in slice_indices[self.config['dataset']]:
            slice_indices[self.config['dataset']][model_name] = {}
        slice_indices[self.config['dataset']][model_name][str(exp_id)] = {}
        slice_indices[self.config['dataset']][model_name][str(exp_id)]['focus_th'] = self.config['focus_th']
        slice_indices[self.config['dataset']][model_name][str(exp_id)]['range_params'] = self.range_params
        slice_indices[self.config['dataset']][model_name][str(exp_id)]['slice_params'] = self.slice_params
        with open(slice_indices_file, 'w') as file:
            json.dump(slice_indices, file, indent=2)
        
        # 构建存储slices的文件(numpy object)
        slices_file = f"./point_slices/{self.config['dataset']}_{model_name}_{str(exp_id)}.npz"
        slices_indices_file = f"./point_slices/indices_{self.config['dataset']}_{model_name}_{str(exp_id)}.npz"
        # local_slices_infor = copy.deepcopy(self.slices_infor)
        # local_slices_infor['data_scope'] = self.data_scope
        
        np.savez(slices_file, metadata_objs=self.metadata_ranges, ranges=self.ranges_infor, slices=self.slices_infor)
        np.savez(slices_indices_file, ranges_indices=self.ranges_indices, slices_indices=self.slices_indices)
        
        return 'ok'