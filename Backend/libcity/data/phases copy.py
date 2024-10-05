import os
import torch
import math
import numpy as np
import pandas as pd
import json

class Phases():
    def __init__(self, config, raw_data, time_strs, geo_coords, phase_params):
        self.config = config
        self.raw_data = raw_data
        self.phase_params = phase_params
        self.focus_th = self.config.get('focus_th', 115) * 0.9
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
    
    def life_span(self, phase, phase_data):
        return phase['end'] - phase['start']
    
    def space_focus_cnt(self, phase, phase_data):
        return 
    
        
    
    def call_method_by_name(self, method_name, phase, phase_data, *args, **kwargs):
        cur_metadata = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            cur_metadata = method(phase, phase_data, *args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return cur_metadata
    
    
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
            phase_infor['phase_id'] = i
            phase_infor['start'] = temporal_seg_points[i]
            # if is_hour_data and phase_infor['start'] % n_step_month == 0:
            #     phase_infor['start'] += input_step_num
            phase_infor['end'] = temporal_seg_points[i+1]
            phase_infor['life_span'] = phase_infor['end'] - phase_infor['start']
            phase_infor['start_date'] = self.time_strs[phase_infor['start']]
            # 保留污染消失的那一天作为污染阶段的最后一天
            phase_infor['end_date'] = self.time_strs[phase_infor['end']-1]
            phase_infor['year_month'] = phase_infor['start_date'][0:5]
            
            
            # 针对每个阶段给出概览的数据信息，focus阶段要更详细一些。
            phase_data = target_val_series[phase_infor['start']:(phase_infor['end'])]
            phase_hist, phase_bin_edges = np.histogram(phase_data.flatten(), bins=self.bin_list, density=False)
            phase_hist_binary, _ = np.histogram(phase_data.flatten(), bins=[0, self.focus_th, 100000], density=False)
            phase_hist = 1.0 * phase_hist / (phase_infor['end'] - phase_infor['start'])
            phase_hist_binary = 1.0 * phase_hist_binary / (phase_hist_binary[0] + phase_hist_binary[1])
            phase_infor['hist'] = phase_hist.tolist()
            phase_infor['hist_binary'] = phase_hist_binary.tolist()
            
            # 统计空间上的污染情况  
            time_focus_val = np.mean(phase_data, axis=1)
            time_focus_event_val = []
            for i in range(phase_data.shape[0]):
                pollu_flags = phase_data[i] > self.focus_th
                if np.count_nonzero(pollu_flags) > 0: 
                    time_focus_event_val.append(float(np.mean(phase_data[i][pollu_flags])))
                else: time_focus_event_val.append(0.0)
            space_focus_cnt = np.mean(phase_data >= self.focus_th, axis=0)
            phase_infor['space_focus_cnt'] = space_focus_cnt.tolist()
            phase_infor['time_focus_val'] = time_focus_val.tolist()
            phase_infor['time_focus_event_val'] = time_focus_event_val
            
            # 统计时间上的污染和非污染情况
            if phase_data.shape[0] > 90:
                down_rate = math.ceil(1.0 * phase_data.shape[0] / 90)
                down_phase_data = phase_data[::down_rate]
                time_focus_cnt = np.mean(down_phase_data >= self.focus_th, axis=1)
            else:
                time_focus_cnt = np.mean(phase_data >= self.focus_th, axis=1)
            time_focus_cnt_all = np.mean(phase_data >= self.focus_th, axis=1)
            phase_infor['time_focus_cnt'] = time_focus_cnt.tolist()
            phase_infor['time_focus_cnt_all'] = time_focus_cnt_all.tolist()
            if phase_infor['type'] == 'focus':
                self.all_phases.append(phase_infor)
        point_locs_array = np.array(self.geo_coords)
        
        # 计算阶段的整体指标
        for phase_id, phase in enumerate(self.all_phases):
            phase['focus_phase_id'] = phase_id
            phase_data = target_val_series[phase['start']:(phase['end'])]
            step_num, loc_num = phase_data.shape
            phase_focus_flag = phase_data >= self.focus_th
            # 计算phase的整体指标
            # 计算每个grid的duration，从而计算阶段的mean duration
            mean_duration = 0
            for i in range(loc_num):
                cur_duration = np.count_nonzero(phase_focus_flag[:,i])
                mean_duration += cur_duration
            mean_duration = 1.0 * mean_duration / loc_num
            # 计算阶段的污染网格数量信息
            accu_grid_num = int(np.count_nonzero(phase_focus_flag))
            mean_grid_num = accu_grid_num / step_num
            focus_counts = np.sum(phase_focus_flag, axis=1)
            max_grid_num = int(np.max(focus_counts))
            # 计算阶段的污染程度信息
            focus_data = phase_data[phase_focus_flag]
            mean_intensity = float(np.mean(focus_data))
            max_intensity = float(np.max(focus_data))
            # 计算阶段的污染移动信息，移动距离、移动方向、移动速度
            # 移动需要计算每一天的污染质心，通过污染强度对经纬度进行加权
            step_centroids = []
            step_centroids_pure = []
            for i in range(step_num):
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
                step_centroids_pure.append(step_centroid.tolist())
            move_dis = 0
            jump_step_num = 0
            for i in range(1, len(step_centroids)):
                if len(step_centroids[i]) == 0 or len(step_centroids[i-1]) == 0: 
                    jump_step_num += 1
                    continue
                cur_dis = self.haversine_distance(step_centroids[int(i)][0], step_centroids[int(i)][1], step_centroids[int(i-1)][0], step_centroids[int(i-1)][1])
                move_dis += cur_dis
            move_speed = move_dis / (phase['life_span'] - jump_step_num)
            
            phase['mean_duration'] = mean_duration
            phase['accumulate_area'] = accu_grid_num
            phase['mean_focus_grids'] = mean_grid_num
            phase['max_focus_grids'] = max_grid_num
            phase['mean_intensity'] = mean_intensity
            phase['max_intensity'] = max_intensity
            phase['step_centroids'] = step_centroids
            phase['move_distance'] = move_dis
            phase['move_speed'] = move_speed
            
        return 'ok'

    def save_phases(self):
        phases_file = f"./phases_data/{self.dataset}_{float(self.config['focus_th'])}_{self.phase_params['min_len']}_{self.phase_params['max_gap_len']}.npy"
        np.save(phases_file, self.all_phases, allow_pickle=True)
        
        return 'ok'