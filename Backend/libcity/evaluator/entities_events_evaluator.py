import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
import numpy as np
import math
import random
import copy
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
from scipy.spatial import distance


class EntitiesEventsEvaluator():
    def __init__(self, config, data_scope, raw_data, truth_series, pred_series, phases, step_entities, events, geo_coords):
        self.config = config
        self.data_scope = data_scope
        self.raw_data = raw_data
        self.truth_series = truth_series
        self.pred_series = pred_series
        self.focus_th = self.config.get('focus_th', '')
        self.phases = phases
        self.step_entities = step_entities
        self.events = events
        self.geo_coords = geo_coords
        
        self.all_pred_entities = []
        self.all_evaluated_pairs = []
    
    def get_connected_components(self, dis_mtx, dis_th, min_grid_num):
        db_clustering = DBSCAN(eps=dis_th, min_samples=min_grid_num,  metric='precomputed').fit(dis_mtx)
        labels = db_clustering.labels_
        # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        connected_components = []
        for label in np.unique(labels):
            if label != -1:  # -1 表示噪声点
                component = np.where(labels == label)[0].tolist()
                connected_components.append(component)
        
        return connected_components
    
    def get_predict_entities(self):
        point_locs_array = np.array(self.geo_coords)
        min_grid_num = int(self.config['event_params']['min_grid_num'])
        connected_dis_th = float(self.config['event_params']['connected_dis_th'])
        for i in range(len(self.pred_series)):
            self.all_pred_entities.append([])
            for j in range(len(self.pred_series[i][0])):
                space_vals = [row[j] for row in self.pred_series[i]]
                step_focus_num = np.count_nonzero(space_vals)
                step_focus_entity = []
                self.all_pred_entities[i].append([])
                if step_focus_num == 0: 
                    # self.all_pred_entities[i][j].append([])
                    continue
                space_focus_flag = np.array(space_vals) > self.focus_th
                step_focus_loc_ids = np.where(space_focus_flag)[0].astype(int)
                step_focus_locs = point_locs_array[space_focus_flag]
                if step_focus_loc_ids.size == 0: continue
                dis_mtx = haversine_distances(np.radians(step_focus_locs)) * 6371000 / 1000
                step_entities = self.get_connected_components(dis_mtx, connected_dis_th, min_grid_num)
                # compute entities' attributes
                for entity in step_entities:
                    entity_obj = {}
                    entity_obj['step_id'] = self.config['input_window'] + i
                    entity_obj['forecast_step'] = j
                    entity_obj['loc_ids'] = step_focus_loc_ids[entity].tolist()
                    entity_obj['area'] = len(entity_obj['loc_ids'])
                    entity_focus_data = [self.pred_series[i][k][j] for k in entity_obj['loc_ids']]
                    entity_obj['max_val'] = float(np.max(entity_focus_data))
                    entity_obj['intensity'] = float(np.mean(entity_focus_data))
                    grid_weights = entity_focus_data / np.sum(entity_focus_data)
                    focus_locs = point_locs_array[entity_obj['loc_ids']]
                    step_centroid = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                    entity_obj['centroid'] = step_centroid.tolist()
                    step_focus_entity.append(entity_obj)
                self.all_pred_entities[i][j] = step_focus_entity
        
        return 'ok'
    
    def flatten_list(self, nested_list):
        flattened = []
        for element in nested_list:
            if isinstance(element, list):
                flattened.extend(self.flatten_list(element))
            else:
                flattened.append(element)
        return flattened
    
    def call_method_by_name(self, method_name, *args, **kwargs):
        cur_metadata = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            cur_metadata = method(*args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return cur_metadata
    
    def process_false_alarm(self, pred_entity, series_step, forecast_step, truth_entity_ids, pred_entity_ids):
        # 构造一个truth_entity
        truth_entity = {}
        truth_entity['area'] = 0
        truth_entity['loc_ids'] = pred_entity['loc_ids']
        truth_entity['step_id'] = series_step + self.config['input_window']
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0].tolist()
        truth_entity['intensity'] = float(np.mean(truth_entity['data']))
        truth_entity['centroid'] = None
        pred_entity['data'] = [self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']]
        pred_entity['forecast_step'] = forecast_step
        eval_obj = {}
        eval_obj['type'] = 'False Alarm'
        eval_obj['series_step'] = series_step
        eval_obj['forecast_step'] = forecast_step
        eval_obj['truth_entity_ids'] = truth_entity_ids
        eval_obj['pred_entity_ids'] = pred_entity_ids
        for metric in self.config['event_params']['error_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        return eval_obj
    
    def process_miss(self, truth_entity, series_step, forecast_step, truth_entity_ids, pred_entity_ids):
        # 构造一个pred_entity
        pred_entity = {}
        pred_entity['area'] = 0
        pred_entity['forecast_step'] = forecast_step
        pred_entity['loc_ids'] = truth_entity['loc_ids']
        pred_entity['data'] = [self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']]
        pred_entity['intensity'] = float(np.mean(pred_entity['data']))
        pred_entity['centroid'] = None
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0].tolist()
        truth_entity['step_id'] = series_step + self.config['input_window']
        eval_obj = {}
        eval_obj['type'] = 'Miss'
        eval_obj['series_step'] = series_step
        eval_obj['forecast_step'] = forecast_step
        eval_obj['truth_entity_ids'] = truth_entity_ids
        eval_obj['pred_entity_ids'] = pred_entity_ids
        for metric in self.config['event_params']['error_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        return eval_obj
    
    def process_match(self, truth_entity, pred_entity, series_step, forecast_step, truth_entity_ids, pred_entity_ids):
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0].tolist()
        truth_entity['step_id'] = series_step + self.config['input_window']
        pred_entity['data'] = [self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']]
        pred_entity['forecast_step'] = forecast_step
        eval_obj = {}
        for metric in self.config['event_params']['error_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        eval_obj['type'] = self.judge_hits_misses(eval_obj)
        eval_obj['series_step'] = series_step
        eval_obj['forecast_step'] = forecast_step
        eval_obj['truth_entity_ids'] = truth_entity_ids
        eval_obj['pred_entity_ids'] = pred_entity_ids
        return eval_obj
    
    def haversine_distance(self, coord1, coord2):
        lat1 = coord1[1]
        lon1 = coord1[0]
        lat2 = coord2[1]
        lon2 = coord2[0]
        # 将经纬度从度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # 地球半径（单位：千米）
        R = 6371.0
        
        # 计算距离
        distance = R * c
        return distance
    
    def compute_entity_match(self, entity1, entity2):
        related_dis_th = float(self.config['event_params']['related_dis_th'])
        loc_ids_1 = entity1['loc_ids']
        loc_ids_2 = entity2['loc_ids']
        if set(loc_ids_1).intersection(set(loc_ids_2)): return 1
        for loc1 in loc_ids_1:
            for loc2 in loc_ids_2:
                cur_dis = self.haversine_distance(self.geo_coords[loc1], self.geo_coords[loc2])
                if cur_dis < related_dis_th:
                    return 1
        return 0
    
    # def compute_truth_entities_errors(self):
    #     point_locs_array = np.array(self.geo_coords)
    #     for i, phase in enumerate(self.phases):
    #         entity_metrics = []
    #         series_start = phase['start'] - self.config['input_window']
    #         series_end = phase['end'] - self.config['input_window']
    #         for j in range(series_start, series_end+1):
    #             step_id = j - series_start
    #             truth_entities = phase['focus_entities'][step_id]
    #             entity_metrics.append([])
    #             for k in range(len(self.pred_series[j][0]))
    #                 entity_metrics[step_id].append([])
    #                 if len(truth_entities) == 0: continue
    #                 for truth_entity in truth_entities:
    #                     entity_pred = np.array([self.pred_series[j][idx][k] for idx in truth_entity['loc_ids']])
    #                     pred_area = int(np.count_nonzero([entity_pred > self.focus_th]))
    #                     area_error = 
                        
                        
                        
    #                     entity_metrics[step_id][k].append(eval_obj)
                        
                        
    #         phase['entity_metrics'] = entity_metrics
    
    def compute_phase_entities_errors(self):
        self.get_predict_entities()
        point_locs_array = np.array(self.geo_coords)
        for i, phase in enumerate(self.phases):
            entity_metrics = []
            # truth_entities = phase['focus_entities']
            series_start = phase['start'] - self.config['input_window']
            series_end = phase['end'] - self.config['input_window']
            pred_entities_series = self.all_pred_entities[series_start:(series_end+1)]
            for j in range(series_start, series_end+1):
                step_id = j - series_start
                truth_entities = phase['focus_entities'][step_id]
                entity_metrics.append([])
                for k in range(len(self.pred_series[j][0])):
                # for k, pred_entities in enumerate(self.all_pred_entities[j]):
                    entity_metrics[step_id].append([])
                    pred_entities = None
                    if len(self.all_pred_entities[j][k]) == 0:
                        # 所有的truth_entities算为miss
                        for l, truth_entity in enumerate(truth_entities):
                            eval_obj = self.process_miss(truth_entity, j, k, [l], None)
                            entity_metrics[step_id][k].append(eval_obj)
                        continue
                    else:
                        pred_entities = self.all_pred_entities[j][k]
                    
                    if len(truth_entities) == 0: continue
                    else:
                        # 需要对step内所有的pred_entity和truth_entity进行匹配
                        if len(truth_entities) == 1 and len(pred_entities) == 1:
                            truth_entity = truth_entities[0]
                            pred_entity = pred_entities[0]
                            # 直接看是否匹配
                            match_flag = self.compute_entity_match(truth_entity, pred_entity)
                            truth_entity['data'] = self.raw_data[j+self.config['input_window'], truth_entity['loc_ids'], 0].tolist()
                            truth_entity['step_id'] = j + self.config['input_window']
                            pred_entity['data'] = [self.pred_series[j][n][k] for n in pred_entity['loc_ids']]
                            if match_flag == 1:
                                eval_obj = self.process_match(truth_entity, pred_entity, j, k, [0], [0])
                                entity_metrics[step_id][k].append(eval_obj)
                            else:
                                eval_obj_miss = self.process_miss(truth_entity, j, k, [0], None)
                                entity_metrics[step_id][k].append(eval_obj_miss)
                        else:
                            match_mtx = np.zeros((len(truth_entities), len(pred_entities)))
                            for m, truth_entity in enumerate(truth_entities):
                                for n, pred_entity in enumerate(pred_entities):
                                    match_mtx[m][n] = self.compute_entity_match(truth_entity, pred_entity)
                            
                            # 统计truth_entity和pred_entity作为节点的度数
                            truth_entity_matches = [[] for _ in range(len(truth_entities))]
                            pred_entity_matches = [[] for _ in range(len(pred_entities))]
                            for m in range(match_mtx.shape[0]):
                                for n in range(match_mtx.shape[1]):
                                    if match_mtx[m][n] == 1:
                                        truth_entity_matches[m].append(n)
                            # 如果两个truth entity所匹配的pred entity有交集的话，则合并相应的truth entity和pred entity
                            truth_eval_cnts = np.zeros(len(truth_entities))
                            pred_eval_cnts = np.zeros(len(pred_entities))
                            
                            # 对pred_entity_matches排序，同时也对pred_entities排序
                            sorted_indices = sorted(range(len(pred_entity_matches)), key=lambda n: len(pred_entity_matches[n]))
                            sorted_pred_matches = [pred_entity_matches[n] for n in sorted_indices]

                            # 判断miss
                            for m in range(len(truth_entity_matches)):
                                if len(truth_entity_matches[m]) == 0:  # miss
                                    eval_obj = self.process_miss(truth_entities[m], j, k, [m], None)
                                    entity_metrics[step_id][k].append(eval_obj)
                            
                            for m in range(len(sorted_pred_matches)):
                                if len(pred_entity_matches[m]) == 0: continue
                                elif len(pred_entity_matches[m]) == 1:
                                    truth_entity_id = pred_entity_matches[m][0]
                                    if truth_eval_cnts[truth_entity_id] > 0: continue
                                    truth_eval_cnts[truth_entity_id] += 1
                                    truth_entity = truth_entities[truth_entity_id]
                                    pred_entity_ids = truth_entity_matches[truth_entity_id]
                                    # 合并pred entities
                                    agg_pred_entity = {}
                                    agg_pred_entity['loc_ids'] = []
                                    for pred_entity_id in pred_entity_ids:
                                        agg_pred_entity['loc_ids'] += pred_entities[pred_entity_id]['loc_ids']
                                    agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                    agg_pred_entity['data'] = [self.pred_series[j][n][k] for n in agg_pred_entity['loc_ids']]
                                    agg_pred_entity['intensity'] = float(np.mean(agg_pred_entity['data']))
                                    agg_pred_entity['forecast_step'] = k
                                    grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                    focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                    agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                    agg_pred_entity['centroid'] = [float(coord) for coord in agg_pred_entity['centroid']]
                                    eval_obj = self.process_match(truth_entity, agg_pred_entity, j, k, [truth_entity_id], pred_entity_ids)
                                    entity_metrics[step_id][k].append(eval_obj)
                                elif len(pred_entity_matches[m]) > 1:
                                    # 先合并truth entities
                                    truth_entity_ids = pred_entity_matches[m]
                                    if np.all(truth_eval_cnts[truth_entity_ids] > 0): continue
                                    pred_entity_ids = []
                                    agg_truth_entity = {}
                                    agg_truth_entity['loc_ids'] = []
                                    for truth_entity_id in truth_entity_ids:
                                        truth_eval_cnts[truth_entity_id] += 1
                                        agg_truth_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                        pred_entity_ids += truth_entity_matches[truth_entity_id]
                                    agg_truth_entity['area'] = len(agg_truth_entity['loc_ids'])
                                    agg_truth_entity['data'] = self.raw_data[j+self.config['input_window'], agg_truth_entity['loc_ids'], 0].tolist()
                                    agg_truth_entity['step_id'] = self.config['input_window'] + j
                                    agg_truth_entity['intensity'] = float(np.mean(agg_truth_entity['data']))
                                    grid_weights = agg_truth_entity['data'] / np.sum(agg_truth_entity['data'])
                                    focus_locs = point_locs_array[agg_truth_entity['loc_ids']]
                                    agg_truth_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                    agg_pred_entity['centroid'] = [float(coord) for coord in agg_pred_entity['centroid']]
                                    # 再合并相应的pred entities
                                    pred_entity_ids = list(set(pred_entity_ids))
                                    agg_pred_entity = {}
                                    agg_pred_entity['loc_ids'] = []
                                    for truth_entity_id in truth_entity_ids:
                                        agg_pred_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                        pred_entity_ids += truth_entity_matches[truth_entity_id]
                                    agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                    agg_pred_entity['data'] = [self.pred_series[j][n][k] for n in agg_pred_entity['loc_ids']]
                                    agg_pred_entity['intensity'] = float(np.mean(agg_pred_entity['data']))
                                    grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                    focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                    agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                    eval_obj = self.process_match(agg_truth_entity, agg_pred_entity, j, k, truth_entity_ids, pred_entity_ids)
                                    entity_metrics[step_id][k].append(eval_obj)
            phase['entity_metrics'] = entity_metrics
    
    def compute_entities_errors(self):
        self.get_predict_entities()
        point_locs_array = np.array(self.geo_coords)
        # match the entities in predictions and ground truth
        # first compute the errors for entities in ground truth
        truth_entities_series = [[] for _ in range(len(self.all_pred_entities))]
        
        for truth_entities in self.step_entities:
            if len(truth_entities) == 0: continue
            series_id = truth_entities[0]['step_id'] - self.config['input_window']
            truth_entities_series[series_id] = truth_entities
        for i, truth_entities in enumerate(truth_entities_series):
            step_pred_entities = self.all_pred_entities[i]
            for j, pred_entities in enumerate(step_pred_entities):
                if len(truth_entities) == 0 and len(pred_entities) == 0: continue
                elif len(truth_entities) == 0:
                    # 所有的pred_entities算为false alarm
                    for k, pred_entity in enumerate(pred_entities):
                        eval_obj = self.process_false_alarm(pred_entity, i, j, None, [k])
                        self.all_evaluated_pairs.append(eval_obj)
                elif len(pred_entities) == 0:
                    # 所有的truth_entities算为miss
                    for k, truth_entity in enumerate(truth_entities):
                        eval_obj = self.process_miss(truth_entity, i, j, [k], None)
                        self.all_evaluated_pairs.append(eval_obj)
                else:
                    # 需要对step内所有的pred_entity和truth_entity进行匹配
                    if len(truth_entities) == 1 and len(pred_entities) == 1:
                        truth_entity = truth_entities[0]
                        pred_entity = pred_entities[0]
                        # 直接看是否匹配
                        shift_dis = np.linalg.norm(np.array(truth_entity['centroid']) - np.array(pred_entity['centroid']))
                        truth_entity['data'] = self.raw_data[i+self.config['input_window'], truth_entity['loc_ids'], 0].tolist()
                        truth_entity['step_id'] = i + self.config['input_window']
                        pred_entity['data'] = [self.pred_series[i][k][j] for k in pred_entity['loc_ids']]
                        if shift_dis < self.config['event_params']['related_dis_th']:
                            eval_obj = self.process_match(truth_entity, pred_entity, i, j, [0], [0])
                            self.all_evaluated_pairs.append(eval_obj)
                        else:
                            eval_obj_far = self.process_false_alarm(pred_entity, i, j, None, [0])
                            eval_obj_miss = self.process_miss(truth_entity, i, j, [0], None)
                            self.all_evaluated_pairs.append(eval_obj_far)
                            self.all_evaluated_pairs.append(eval_obj_miss)
                    else:
                        truth_centroids = [entity['centroid'] for entity in truth_entities]
                        pred_centroids = [entity['centroid'] for entity in pred_entities]
                        distances = distance.cdist(truth_centroids, pred_centroids)
                        # 统计truth_entity和pred_entity作为节点的度数
                        truth_entity_matches = [[] for _ in range(len(truth_entities))]
                        pred_entity_matches = [[] for _ in range(len(pred_entities))]
                        matched_pairs = []
                        for k in range(distances.shape[0]):
                            for l in range(distances.shape[1]):
                                if distances[k][l] < self.config['event_params']['related_dis_th']:
                                    truth_entity_matches[k].append(l)
                                    pred_entity_matches[l].append(k)
                                    matched_pairs.append([k,l])
                        # 如果两个truth entity所匹配的pred entity有交集的话，则合并相应的truth entity和pred entity
                        truth_eval_cnts = np.zeros(len(truth_entities))
                        pred_eval_cnts = np.zeros(len(pred_entities))
                        # 对pred_entity_matches排序，同时也对pred_entities排序
                        sorted_indices = sorted(range(len(pred_entity_matches)), key=lambda i: len(pred_entity_matches[i]))
                        sorted_pred_matches = [pred_entity_matches[i] for i in sorted_indices]
                        # sorted_pred_entities = [pred_entities[i] for i in sorted_indices]
                        # 判断miss
                        for k in range(len(truth_entity_matches)):
                            if truth_entity_matches[k] == 0:  # miss
                                eval_obj = self.process_miss(truth_entities[k], i, j, [k], None)
                                self.all_evaluated_pairs.append(eval_obj)
                        for k in range(len(sorted_pred_matches)):
                            if len(pred_entity_matches[k]) == 0:  # false alarm
                                eval_obj = self.process_false_alarm(pred_entities[k], i, j, None, [k])
                                self.all_evaluated_pairs.append(eval_obj)
                            elif len(pred_entity_matches[k]) == 1:
                                truth_entity_id = pred_entity_matches[k][0]
                                if truth_eval_cnts[truth_entity_id] > 0: continue
                                truth_eval_cnts[truth_entity_id] += 1
                                truth_entity = truth_entities[truth_entity_id]
                                pred_entity_ids = truth_entity_matches[truth_entity_id]
                                # 合并pred entities
                                agg_pred_entity = {}
                                agg_pred_entity['loc_ids'] = []
                                for pred_entity_id in pred_entity_ids:
                                    agg_pred_entity['loc_ids'] += pred_entities[pred_entity_id]['loc_ids']
                                agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                agg_pred_entity['data'] = [self.pred_series[i][k][j] for k in agg_pred_entity['loc_ids']]
                                agg_pred_entity['intensity'] = np.mean(agg_pred_entity['data'])
                                agg_pred_entity['forecast_step'] = j
                                grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                eval_obj = self.process_match(truth_entity, agg_pred_entity, i, j, [truth_entity_id], pred_entity_ids)
                                self.all_evaluated_pairs.append(eval_obj)
                            elif len(pred_entity_matches[k]) > 1:
                                # 先合并truth entities
                                truth_entity_ids = pred_entity_matches[k]
                                if np.all(truth_eval_cnts[truth_entity_ids] > 0): continue
                                pred_entity_ids = []
                                agg_truth_entity = {}
                                agg_truth_entity['loc_ids'] = []
                                for truth_entity_id in truth_entity_ids:
                                    truth_eval_cnts[truth_entity_id] += 1
                                    agg_truth_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                    pred_entity_ids += truth_entity_matches[truth_entity_id]
                                agg_truth_entity['area'] = len(agg_truth_entity['loc_ids'])
                                agg_truth_entity['data'] = self.raw_data[i+self.config['input_window'], agg_truth_entity['loc_ids'], 0].tolist()
                                agg_truth_entity['step_id'] = self.config['input_window'] + i
                                agg_truth_entity['intensity'] = np.mean(agg_truth_entity['data'])
                                grid_weights = agg_truth_entity['data'] / np.sum(agg_truth_entity['data'])
                                focus_locs = point_locs_array[agg_truth_entity['loc_ids']]
                                agg_truth_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                # 再合并相应的pred entities
                                pred_entity_ids = list(set(pred_entity_ids))
                                agg_pred_entity = {}
                                agg_pred_entity['loc_ids'] = []
                                for truth_entity_id in truth_entity_ids:
                                    agg_pred_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                    pred_entity_ids += truth_entity_matches[truth_entity_id]
                                agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                agg_pred_entity['data'] = [self.pred_series[i][k][j] for k in agg_pred_entity['loc_ids']]
                                agg_pred_entity['intensity'] = np.mean(agg_pred_entity['data'])
                                grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0).tolist()
                                eval_obj = self.process_match(agg_truth_entity, agg_pred_entity, i, j, truth_entity_ids, pred_entity_ids)
                                self.all_evaluated_pairs.append(eval_obj)
        
        return 'ok'
    
    
    
    def area_error(self, truth_entity, pred_entity):
        area_change_th = self.config['event_params']['area_change_th']
        if abs(truth_entity['area'] - pred_entity['area']) < truth_entity['area'] * area_change_th:
            return ['Approx. correct', pred_entity['area']-truth_entity['area']]
        elif truth_entity['area'] - pred_entity['area'] > truth_entity['area'] * area_change_th:
            return ['Underestimate', pred_entity['area']-truth_entity['area']]
        elif pred_entity['area'] - truth_entity['area'] > truth_entity['area'] * area_change_th:
            return ['Overestimate', pred_entity['area']-truth_entity['area']]
    
    def intensity_error(self, truth_entity, pred_entity):
        val_change_th = self.config['event_params']['val_change_th']
        if abs(truth_entity['intensity'] - pred_entity['intensity']) < truth_entity['intensity'] * val_change_th:
            return ['Approx. correct', pred_entity['intensity']-truth_entity['intensity']]
        elif truth_entity['intensity'] - pred_entity['intensity'] > truth_entity['intensity'] * val_change_th:
            return ['Underestimate', pred_entity['intensity']-truth_entity['intensity']]
        elif pred_entity['intensity'] - truth_entity['intensity'] > truth_entity['intensity'] * val_change_th:
            return ['Overestimate', pred_entity['intensity']-truth_entity['intensity']]

    def haversine_distance(self, coord1, coord2):
        lat1 = coord1[1]
        lon1 = coord1[0]
        lat2 = coord2[1]
        lon2 = coord2[0]
        # 将经纬度从度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # 地球半径（单位：千米）
        R = 6371.0
        
        # 计算距离
        distance = R * c
        return distance
    
    
    def shift_error(self, truth_entity, pred_entity):
        if truth_entity['centroid'] is None or pred_entity['centroid'] is None:
            return None
        else:
            # return float(np.linalg.norm(np.array(truth_entity['centroid']) - np.array(pred_entity['centroid'])))
            return self.haversine_distance(truth_entity['centroid'], pred_entity['centroid'])

    def union_points_error(self, truth_entity, pred_entity):
        union_array = np.union1d(np.array(truth_entity['loc_ids']), np.array(pred_entity['loc_ids']))
        union_truth = self.raw_data[truth_entity['step_id'], union_array, 0]
        
        union_pred = np.array([self.pred_series[truth_entity['step_id']-self.config['input_window']][k][pred_entity['forecast_step']] for k in union_array])
        return float(loss.masked_mae_np(union_truth, union_pred))
    
    def judge_hits_misses(self, eval_obj):
        # 在match的情况下，判断是否是hit，或者是miss，或者也可能是False Alarm
        # 或者换一组名字，hit，over，under
        # 根据intensity和area的情况，给一个标记，便于筛选和查看
        intensity_error = eval_obj['intensity_error']
        area_error = eval_obj['area_error']
        
        if intensity_error == 'Approx. correct' and area_error == 'Approx. correct':
            return 'Hit-Hit'
        if intensity_error == 'Approx. correct' and area_error == 'Overestimate':
            return 'Hit-Overestimate'
        if intensity_error == 'Approx. correct' and area_error == 'Underestimate':
            return 'Hit-Underestimate'
        if intensity_error == 'Overestimate' and area_error == 'Approx. correct':
            return 'Overestimate-Hit'
        if intensity_error == 'Overestimate' and area_error == 'Overestimate':
            return 'Overestimate-Overestimate'
        if intensity_error == 'Overestimate' and area_error == 'Underestimate':
            return 'Overestimate-Underestimate'
        if intensity_error == 'Underestimate' and area_error == 'Approx. correct':
            return 'Underestimate-Hit'
        if intensity_error == 'Underestimate' and area_error == 'Overestimate':
            return 'Underestimate-Overestimate'
        if intensity_error == 'Underestimate' and area_error == 'Underestimate':
            return 'Underestimate-Underestimate'
    
    # def compute_entities_errors(self):
    #     for i, phase in enumerate(self.phases):
    #         for j, event in enumerate(phase['phase_events']):
    #             # (phase_id, i, prev, prev_entity['post_match_ids'], area_obj, val_obj, move_obj)
    #             phase_step = event[1]
    #             series_step = phase['start'] + phase_step
    #             prev_entities = event[2]
    #             post_entities = event[3]
    #             area_obj = event[4]
    #             val_obj = event[5]
    #             move_obj = event[6]
    #             eval_obj = {}
                
    #             for k in 