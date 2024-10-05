import os
import torch
import math
import numpy as np
import pandas as pd
import json
import hashlib

class Phases():
    def __init__(self, config, raw_data, model_name, time_strs, geo_coords, phase_params):
        self.config = config
        self.raw_data = raw_data
        self.model_name = model_name
        self.phase_params = phase_params
        self.focus_th = self.config.get('focus_th', 115)
        self.dataset = self.config.get('dataset', '')
        self.input_window = self.config.get('input_window', '')
        self.min_length = phase_params['min_len']
        self.max_gap_len = phase_params['max_gap_len']
        self.bin_list = self.config.get('focus_levels', '')
        self.time_strs = time_strs
        self.geo_coords = geo_coords
        self.all_phases = []
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        R = 6371.0
        distance = R * c
        return distance
    
    def life_span(self, phase_infor, phase_data):
        return phase_infor['end'] - phase_infor['start']
    
    def space_focus_cnt(self, phase_infor, phase_data):
        return  np.mean(phase_data >= self.focus_th, axis=0).tolist()

    def time_val(self, phase_infor, phase_data):
        return np.mean(phase_data, axis=1).tolist()
    
    def time_focus_val(self, phase_infor, phase_data):
        time_focus_vals = []
        for i in range(phase_data.shape[0]):
            focus_flags = phase_data[i] > self.focus_th
            if np.count_nonzero(focus_flags) > 0: 
                time_focus_vals.append(float(np.mean(phase_data[i][focus_flags])))
            else: time_focus_vals.append(0.0)
        return time_focus_vals

    # def time_focus_cnt_compact(self, phase_infor, phase_data):
    #     if phase_data.shape[0] > 90:
    #         down_rate = math.ceil(1.0 * phase_data.shape[0] / 90)
    #         down_phase_data = phase_data[::down_rate]
    #         time_focus_cnt = np.mean(down_phase_data >= self.focus_th, axis=1)
    #     else:
    #         time_focus_cnt = np.mean(phase_data >= self.focus_th, axis=1)
    #     return time_focus_cnt.tolist()
    
    def time_focus_cnt(self, phase_infor, phase_data):
        return np.mean(phase_data >= self.focus_th, axis=1).tolist()
    
    def focus_level_hist(self, phase_infor, phase_data):
        phase_hist, _ = np.histogram(phase_data.flatten(), bins=self.bin_list, density=False)
        phase_hist = 1.0 * phase_hist / (phase_infor['end'] - phase_infor['start'])
        return phase_hist.tolist()
    
    def mean_duration(self, phase_infor, phase_data):
        mean_duration = 0
        phase_focus_flag = phase_data >= self.focus_th
        for i in range(phase_data.shape[1]):
            cur_duration = np.count_nonzero(phase_focus_flag[:,i])
            mean_duration += cur_duration
        mean_duration = 1.0 * mean_duration / phase_data.shape[1]
        return mean_duration
    
    def mean_focus_grids(self, phase_infor, phase_data):
        accu_grid_num = int(np.count_nonzero(phase_data >= self.focus_th))
        mean_grid_num = accu_grid_num / phase_data.shape[0]
        return mean_grid_num
    
    def max_step_focus_grids(self, phase_infor, phase_data):
        time_focus_num = np.sum(phase_data >= self.focus_th, axis=1)
        max_grid_num = int(np.max(time_focus_num))
        return max_grid_num
        
    def mean_intensity(self, phase_infor, phase_data):
        focus_data = phase_data[phase_data >= self.focus_th]
        return float(np.mean(focus_data))
        
    def max_step_intensity(self, phase_infor, phase_data):
        if 'time_focus_val' in phase_infor:
            return float(np.max(phase_infor['time_focus_val']))
        time_focus_vals = []
        for i in range(phase_data.shape[0]):
            focus_flags = phase_data[i] > self.focus_th
            if np.count_nonzero(focus_flags) > 0: 
                time_focus_vals.append(float(np.mean(phase_data[i][focus_flags])))
            else: time_focus_vals.append(0.0)
        return float(np.max(time_focus_vals))
    
    def max_value(self, phase_infor, phase_data):
        return float(np.max(phase_data))
    
    def step_centroids(self, phase_infor, phase_data):
        phase_focus_flag = phase_data >= self.focus_th
        point_locs_array = np.array(self.geo_coords)
        step_centroids = []
        for i in range(phase_data.shape[0]):
            # 计算各个地点的权重
            step_focus_num = np.count_nonzero(phase_focus_flag[i])
            if step_focus_num == 0: 
                step_centroids.append([])
                continue
            step_pullution_data = phase_data[i][phase_focus_flag[i]]
            grid_weights = step_pullution_data / np.sum(step_pullution_data)
            focus_locs = point_locs_array[phase_focus_flag[i]]
            step_centroid = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
            step_centroids.append(step_centroid.tolist())
        return step_centroids
        
    def mean_move_distance(self, phase_infor, phase_data):
        if 'step_centroids' in phase_infor:
            step_centroids = phase_infor['step_centroids']
        else:
            step_centroids = self.step_centroids(phase_infor, phase_data)
        move_dis = 0
        jump_step_num = 0
        for i in range(1, len(step_centroids)):
            if len(step_centroids[i]) == 0 or len(step_centroids[i-1]) == 0: 
                jump_step_num += 1
                continue
            cur_dis = self.haversine_distance(step_centroids[int(i)][0], step_centroids[int(i)][1], step_centroids[int(i-1)][0], step_centroids[int(i-1)][1])
            move_dis += cur_dis
        mean_move_dis = move_dis / (phase_data.shape[0] - jump_step_num)
        return mean_move_dis
    
    def call_method_by_name(self, method_name, phase_infor, phase_data, *args, **kwargs):
        attr_val = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            attr_val = method(phase_infor, phase_data, *args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return attr_val
    
    def get_all_phases(self):
        target_val_series = self.raw_data[:,:,0]
        flag_array = np.all(target_val_series < self.focus_th, axis=1)
        temporal_seg_points = [self.input_window]
        flag_array[:self.input_window] = True
        # 寻找阶段分割点
        for i in range(self.input_window, target_val_series.shape[0]-1):
            if i < self.input_window:
                continue
            seg_flag = True
            if flag_array[i] == flag_array[i-1]:
                seg_flag = False
            else:
                for j in range(1, int(self.max_gap_len)):
                    if flag_array[i-1] == flag_array[i+j]:
                        seg_flag = False
                        flag_array[i:i+j] = flag_array[i-1]
                        break
            if seg_flag:
                # print(i, flag_array[i])
                temporal_seg_points.append(i)
        temporal_seg_points.append(target_val_series.shape[0])
        # 计算阶段的信息
        for i in range(len(temporal_seg_points)-1):
            if temporal_seg_points[i+1] - temporal_seg_points[i] < self.min_length:
                continue
            phase_infor = {}
            # 阶段的类型（no_focus or focus）
            if flag_array[temporal_seg_points[i]]:
                phase_infor['type'] = 'no_focus'
                continue
            else:
                phase_infor['type'] = 'focus'
            # 阶段的起始
            phase_infor['start'] = temporal_seg_points[i]
            # if is_hour_data and phase_infor['start'] % n_step_month == 0:
            #     phase_infor['start'] += input_step_num
            phase_infor['end'] = temporal_seg_points[i+1]+1
            phase_infor['start_date'] = self.time_strs[phase_infor['start']]
            # 保留污染消失的那一天作为污染阶段的最后一天
            phase_infor['end_date'] = self.time_strs[phase_infor['end']-1]
            phase_infor['year_month'] = phase_infor['start_date'][0:5]
            
            if phase_infor['type'] == 'focus':
                self.all_phases.append(phase_infor)
        
        # 计算阶段的整体指标
        for phase_id, phase in enumerate(self.all_phases):
            phase['focus_phase_id'] = phase_id
            phase_data = target_val_series[phase['start']:(phase['end'])]
            for attr in self.phase_params['attributes']:
                phase[attr] = self.call_method_by_name(attr, phase, phase_data)
                # print(attr, phase['attr'])
        
        return 'ok'

    def save_phases(self):
        def generate_file_label(attributes):
            sorted_attributes = sorted(attributes)
            attributes_str = "_".join(sorted_attributes)
            file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
            file_label = f"{file_hash[:8]}"
            return file_label
        params_label = generate_file_label(self.phase_params['attributes'])
        phases_file = f"./phases_data/{self.dataset}_{self.model_name}_{float(self.config['focus_th'])}_{self.phase_params['min_len']}_{self.phase_params['max_gap_len']}_{params_label}.npy"
        np.save(phases_file, self.all_phases, allow_pickle=True)
        
        return 'ok'