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
from analysis.extract_subset_iterative_point import extract_frequent_subset_iterative

class SliceEvaluator():
    def __init__(self, config, data_scope, metadata, residuals, residual_bins, phases):
        self.config = config
        self.data_scope = data_scope
        self.metadata = metadata
        self.residuals = residuals
        self.residual_bins = residual_bins
        self.phases = phases
        self.range_params = config.get('range_params', {})
        self.slice_params = config.get('slice_params', {})
        self.scope_residuals = []
        self.scope_metadata = {}
        self.metadata_ranges = {}
        self.ranges_infor = {}
        self.slices_infor = {}
        self.global_st_indices_list = []
        self.scope_st_indices_list = []
    
    def flatten_list(self, multi_list):
        return np.array([item for sublist1 in multi_list for sublist2 in sublist1 for item in sublist2])
    
    def process_global_st_indices(self):
        for i in range(len(self.residuals)):
            self.global_st_indices_list.append([])
            for j in range(len(self.residuals[i])):
                self.global_st_indices_list[i].append([])
                for k in range(len(self.residuals[i][j])):
                    self.global_st_indices_list[i][j].append([i,j,k])
    
    def process_scope_data(self):
        step_range = self.slice_params['focus_step_range']
        if self.data_scope == 'All_Data':
            self.scope_residuals = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.residuals])
            for attr in self.metadata.keys():
                self.scope_metadata[attr] = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.metadata[attr]])
            self.scope_st_indices_list = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.global_st_indices_list])
        elif self.data_scope == 'All_Phases':
            phase_ins_ids = []
            for phase_id, phase in enumerate(self.phases):
                step_start = phase['start'] - self.config['input_window']
                step_end = phase['end'] - self.config['input_window']
                phase_ins_ids += list(range(step_start, step_end+1))
            self.scope_residuals = [self.residuals[ins_id] for ins_id in phase_ins_ids]
            self.scope_residuals = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.scope_residuals])
            for attr in self.metadata.keys():
                self.scope_metadata[attr] = [self.metadata[attr][ins_id] for ins_id in phase_ins_ids]
                self.scope_metadata[attr] = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.scope_metadata[attr]])
            phase_st_indices = [self.global_st_indices_list[ins_id] for ins_id in phase_ins_ids]
            self.scope_st_indices_list = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in phase_st_indices])
        elif self.data_scope == 'Selected_Phase':
            phase_id = self.slice_params['phase_id']
            phase = self.phases[phase_id]
            step_start = phase['start'] - self.config['input_window']
            step_end = phase['end'] - self.config['input_window']
            self.scope_residuals = self.residuals[step_start:step_end]
            self.scope_residuals = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.scope_residuals])
            for attr in self.metadata.keys():
                self.scope_metadata[attr] = self.metadata[attr][step_start:step_end]
                self.scope_metadata[attr] = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in self.scope_metadata[attr]])
            phase_st_indices = self.global_st_indices_list[step_start:step_end]
            self.scope_st_indices_list = self.flatten_list([[loc_list[step_range[0]:(step_range[1]+1)] for loc_list in step_list] for step_list in phase_st_indices])
    
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
    
    def range_division(self):
        div_th = self.range_params['div_th']
        div_val_th = div_th * float(np.mean(self.scope_residuals))
        for attr in self.metadata.keys():
            self.metadata_ranges[attr] = {'range': [np.min(self.scope_metadata[attr]), np.max(self.scope_metadata[attr])]}
        if self.range_params['mode'] == 'equal_val':
            bin_num =  int(self.range_params['bin_num'])
            for attr in self.scope_metadata.keys():
                bin_edges = self.divide_range_by_step(self.metadata_ranges[attr]['range'], 1, bin_num)
                self.metadata_ranges[attr]['bin_edges'] = bin_edges
                self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
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
        elif self.range_params['mode'] == 'adaptive':
            bin_num =  int(self.range_params['bin_num'])
            for attr in self.scope_metadata.keys():
                # 获取attr的数据
                attr_data = self.scope_metadata[attr]
                tree = DecisionTreeRegressor(max_leaf_nodes=bin_num)
                tree.fit(attr_data, attr_data)
                bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
                self.metadata_ranges[attr]['bin_edges'] = bin_edges
                self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
        elif self.range_params['mode'] == 'supervised':
            val_step = int(self.range_params['val_step'])
            sup_th = self.range_params['sup_th']
            ranges_infor = seg_range_point(self.metadata_ranges, self.scope_metadata, self.scope_residuals, val_step, sup_th, div_val_th)
            for attr in self.metadata.keys():
                bin_edges = ranges_infor[attr]['seg_points']
                self.metadata_ranges[attr]['bin_edges'] = bin_edges
                self.metadata_ranges[attr]['bins'] = [[bin_edges[i], bin_edges[i+1]] for i in range(len(bin_edges)-1)]
                important_leaf_nodes = ranges_infor[attr]['important_leaf_nodes']
                important_leaf_nodes_range = [node['range'] for node in important_leaf_nodes]
                self.metadata_ranges[attr]['important_bins'] = important_leaf_nodes_range
                self.ranges_infor[attr] = important_leaf_nodes
            
        return 'ok'
    
    def subgroup_mining(self):
        self.process_global_st_indices()
        print("finish process indices")
        self.process_scope_data()
        print("finish process scope_data")
        self.range_division()
        print("finish range division")
        err_diff_val_th = float(self.slice_params['err_diff_th']) * float(np.mean(self.scope_residuals))
        subset_div_val_th = float(self.slice_params['div_th']) * float(np.mean(self.scope_residuals))
        print("prepare to run function")
        scope_subsets_iterative = extract_frequent_subset_iterative(self.ranges_infor, self.scope_residuals, self.scope_metadata, self.metadata_ranges, self.residual_bins, self.slice_params['frequent_sup_th'], err_diff_val_th, subset_div_val_th, self.slice_params['redundancy_th'])
        print('finish computing slices...')
        for i, subset in enumerate(scope_subsets_iterative):
            subset['st_indices'] = [self.scope_st_indices_list[j] for j in subset['indices_list']]
            if 'contain_subsets' in subset:
                for individual_subset in subset['contain_subsets']:
                    individual_subset['global_st_indices'] = [self.scope_st_indices_list[j] for j in individual_subset['indices_list']]
            
        for i, subset in enumerate(scope_subsets_iterative):
            del subset['indices']
            subset['subset_id'] = str(i+1)
            if 'contain_subsets' in subset:
                for j, individual_subset in enumerate(subset['contain_subsets']):
                    del individual_subset['indices']
                    individual_subset['subset_id'] = str(i+1) + '.' + str(j+1)
        
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
        local_slices_infor = copy.deepcopy(self.slices_infor)
        local_slices_infor['data_scope'] = self.data_scope
        np.savez(slices_file, ranges=self.ranges_infor, slices=local_slices_infor)
        
        return 'ok'