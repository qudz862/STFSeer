import os
import torch
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random
import hashlib

# 元属性应该先确定层级：时空位置、输入序列、输出序列、输入&输出序列

class PointMetaData():
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data
        self.variables = list(self.config['dyna']['state'].keys())[1:]
        self.dataset = self.config.get('dataset', '')
        self.output_dim = self.config.get('output_dim', 1)
        self.raw_meta_data = {}
        self.meta_data_instances = {}
        self.meta_data_series = {}

    def target_val(self, var_id, attr_trans):
        if attr_trans == 'abs': return np.abs(self.raw_data[..., var_id])
        else: return self.raw_data[..., var_id]

    def temporal_state_vals(self, var_id, attr_trans):
        target_data = self.raw_data[...,var_id]
        tv = np.zeros_like(target_data)
        tv[1:] = target_data[1:] - target_data[:-1]
        
        if attr_trans == 'abs': return np.abs(tv)
        else: return tv
    
    # def external_state_vals(self):
    #     external_data = self.raw_data[...,1:]
    #     return 
    
    def temporal_context_mean(self, var_id, attr_trans, context_len=6):
        n_step, n_loc, n_var = self.raw_data.shape
        context_means = np.zeros((n_step, n_loc))
        for current_time_index in range(self.config['input_window'], n_step):
            start_index = max(0, current_time_index - context_len + 1)
            context = self.raw_data[start_index:current_time_index, :, var_id]
            context_means[current_time_index] = np.mean(context)
        if attr_trans == 'abs': return np.abs(context_means)
        else: return context_means

    def temporal_context_trend(self, var_id, attr_trans, context_len=6):
        n_step, n_loc, n_var = self.raw_data.shape
        trend_slopes = np.zeros((n_step, n_loc))
        for current_time_index in range(self.config['input_window'], n_step):
            start_index = max(0, current_time_index - context_len + 1)
            context = self.raw_data[start_index:current_time_index,:,var_id]
            context_max = np.max(context)
            context_min = np.min(context)
            X = np.arange(len(context)).reshape(-1, 1)  # 时间点的索引作为特征
            y = context  # 时间序列数据作为目标变量

            model = LinearRegression()
            model.fit(X, y)
        
            # trend_slopes[current_time_index] = model.coef_[0] * (context_max-context_min)
            trend_slopes[current_time_index] = model.coef_[0]
        
        if attr_trans == 'abs': return np.abs(trend_slopes)
        else: return trend_slopes

    def space_comp_state_vals(self, var_id, attr_trans):
        out_data = self.raw_data[...,var_id]
        n_step, n_loc, n_var = self.raw_data.shape
        # 构建反距离权重矩阵
        rel_file = './raw_data/' + self.dataset + '/' + self.dataset + '.rel'
        rel_df = pd.read_csv(rel_file)
        rel_mtx = np.zeros((n_loc, n_loc))
        for i in range(n_loc):
            for j in range(n_loc):
                if i == j: continue
                df_index = i * (n_loc-1) + j
                dis = rel_df.loc[df_index, 'distance']
                rel_mtx[i][j] = 1 / dis
        # 令关系权重和为1
        norm_rel_mtx = np.zeros((n_loc, n_loc))
        for i in range(n_loc):
            rel_sum = np.sum(rel_mtx[i])
            norm_rel_mtx[i] = rel_mtx[i] / rel_sum
        sc = np.zeros((n_step, n_loc))
        for i in range(n_step):
            values_broadcasted = out_data[i][:, np.newaxis]
            weighted_sums = norm_rel_mtx @ out_data[i]
            sc[i] = values_broadcasted.squeeze() - weighted_sums
        
        if attr_trans == 'abs': return np.abs(sc)
        else: return sc

    def space_diff_state_vals(self, var_id, attr_trans):
        out_data = self.raw_data[...,var_id]
        n_step, n_loc, n_var = self.raw_data.shape
        # 构建反距离权重矩阵
        rel_file = './raw_data/' + self.dataset + '/' + self.dataset + '.rel'
        rel_df = pd.read_csv(rel_file)
        rel_mtx = np.zeros((n_loc, n_loc))
        for i in range(n_loc):
            for j in range(n_loc):
                if i == j: continue
                df_index = i * (n_loc-1) + j
                dis = rel_df.loc[df_index, 'distance']
                rel_mtx[i][j] = 1 / dis
        # 令关系权重和为1
        norm_rel_mtx = np.zeros((n_loc, n_loc))
        for i in range(n_loc):
            rel_sum = np.sum(rel_mtx[i])
            norm_rel_mtx[i] = rel_mtx[i] / rel_sum
        sd = np.zeros((n_step, n_loc))
        for i in range(n_step):
            values_broadcasted = out_data[i][:, np.newaxis]
            abs_diff_matrix = np.abs(values_broadcasted - values_broadcasted.T)  # 形状 (n, n)
            sd[i] = np.sum(norm_rel_mtx * abs_diff_matrix, axis=1)  # 形状 (n,)
        
        if attr_trans == 'abs': return np.abs(sd)
        else: return sd
    
    def generate_instances(self, cur_metadata):
        num_samples = cur_metadata.shape[0]
        input_window = self.config['input_window']
        output_window = self.config['output_window']
        output_offset = self.config['output_offset']
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-input_window + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.arange(output_offset, output_window + output_offset, 1))

        y = []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            y_t = cur_metadata[t + y_offsets, ...]
            y.append(y_t)
        y = np.stack(y, axis=0)
        return y

    def generate_step_series(self, cur_metadata):
        n_step, n_loc = cur_metadata.shape
        input_window = self.config['input_window']
        output_window = self.config['output_window']
        output_offset = self.config['output_offset']
        error_step_num = n_step - input_window - (output_offset-1)
        step_val_list = []
        for i in range(error_step_num):
            step_val_list.append([])
            for j in range(n_loc):
                step_val_list[i].append([])
        for i in range(error_step_num):
            pred_num = 0
            if i < output_window:
                pred_num = i + 1
            elif i > error_step_num - output_window:
                pred_num = error_step_num - i
            else: pred_num = output_window
            for j in range(pred_num):
                raw_step_id = i + j + input_window + (output_offset-1)
                if raw_step_id >= n_step: continue
                for k in range(n_loc):
                    step_val_list[i][k].append(cur_metadata[raw_step_id][k])
        return np.array(step_val_list, dtype=object)

    def call_method_by_name(self, method_name, var_id, attr_trans, *args, **kwargs):
        cur_metadata = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            cur_metadata = method(var_id, attr_trans, *args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return cur_metadata
    
    def get_metadata_by_type(self):
        # 先获取raw data的metadata
        # 根据raw data的metadata，获取样本序列和step形式序列的metadata
        for attr in self.config['point_metadata']:
            attr_var, attr_type, attr_trans = None, None, None
            split_attr = attr.split('-')
            if len(split_attr) == 3: attr_var, attr_type, attr_trans = split_attr
            if len(split_attr) == 2: attr_var, attr_type = split_attr
            var_id = self.variables.index(attr_var)
            cur_metadata = self.call_method_by_name(attr_type, var_id, attr_trans)
            cur_metadata_ins = self.generate_instances(cur_metadata)
            cur_metadata_series = self.generate_step_series(cur_metadata)
            self.raw_meta_data[attr] = cur_metadata
            self.meta_data_instances[attr] = cur_metadata_ins
            self.meta_data_series[attr] = cur_metadata_series
        
        return 'ok'
    
    # def get_metadata_saliency(self):
        # 针对meta_data_instances或meta_data_series构建样本
    
    def compute_point_metadata(self):
        input_window = self.config['input_window']
        output_window = self.config['output_window']
        output_offset = self.config['output_offset']
        
        def generate_file_label(attributes):
            sorted_attributes = sorted(attributes)
            attributes_str = "_".join(sorted_attributes)
            file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
            file_label = f"{file_hash[:8]}"
            return file_label
        
        # 先判断元数据文件是否已存在
        metadata_dir = f'./meta_data/{self.dataset}/'
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)
        params_label = generate_file_label(self.config['point_metadata'])
        metadata_raw_file = f'{metadata_dir}/point_metadata_raw_{self.dataset}_{params_label}.npz'
        metadata_ins_file = f'{metadata_dir}/point_metadata_instances_{self.dataset}_{params_label}_{input_window}_{output_window}_{output_offset}.npz'
        metadata_series_file = f'{metadata_dir}/point_metadata_series_{self.dataset}_{params_label}_{input_window}_{output_window}_{output_offset}.npz'
        
        if not self.config['compute_metadata'] and os.path.exists(metadata_raw_file):
            print("Point metadata already existed.")
            self.raw_meta_data = np.load(metadata_raw_file, allow_pickle=True)
            self.meta_data_instances = np.load(metadata_ins_file, allow_pickle=True)
            self.meta_data_series = np.load(metadata_series_file, allow_pickle=True)
            print("Point metadata have been loaded.")
            return 'ok'
        
        self.get_metadata_by_type()
        
        # 将3类元属性分别存为文件
        np.savez(metadata_raw_file, **self.raw_meta_data)
        np.savez(metadata_ins_file, **self.meta_data_instances)
        np.savez(metadata_series_file, **self.meta_data_series)
        print("Point metadata have been computed and saved.")
        
        return 'ok'