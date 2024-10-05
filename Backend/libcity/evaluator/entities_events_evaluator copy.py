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
        min_grid_num = int(self.event_params['min_grid_num'])
        connected_dis_th = float(self.event_params['connected_dis_th'])
        for i in range(len(self.pred_series)):
            self.all_pred_entities.append([])
            for j in range(len(self.pred_series[i][0])):
                space_vals = [row[j] for row in self.pred_series[i]]
                step_focus_num = np.count_nonzero(space_vals)
                step_focus_entity = []
                if step_focus_num == 0: 
                    self.all_pred_entities[i].append([])
                    continue
                space_focus_flag = np.array(space_vals) > self.focus_th
                step_focus_loc_ids = np.where(space_focus_flag)[0].astype(int)
                step_focus_locs = point_locs_array[space_focus_flag]
                dis_mtx = haversine_distances(np.radians(step_focus_locs)) * 6371000 / 1000
                step_entities = self.get_connected_components(dis_mtx, connected_dis_th, min_grid_num)
                # compute entities' attributes
                for entity in step_entities:
                    entity_obj = {}
                    entity_obj['step_id'] = self.config['input_window'] + i
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
                self.all_pred_entities[i].append(step_focus_entity)
            
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
    
    def process_false_alarm(self, pred_entity, series_step, forecast_step):
        # 构造一个truth_entity
        truth_entity = {}
        truth_entity['area'] = 0
        truth_entity['loc_ids'] = pred_entity['loc_ids']
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0]
        truth_entity['intensity'] = np.mean(truth_entity['data'])
        truth_entity['centroid'] = None
        pred_entity['data'] = np.array([self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']])
        eval_obj = {}
        eval_obj['type'] = 'False Alarm'
        for metric in self.config['entity_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        return eval_obj
    def process_miss(self, truth_entity, series_step, forecast_step):
        # 构造一个pred_entity
        pred_entity = {}
        pred_entity['area'] = 0
        pred_entity['loc_ids'] = truth_entity['loc_ids']
        pred_entity['data'] = np.array([self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']])
        pred_entity['intensity'] = np.mean(pred_entity['data'])
        pred_entity['centroid'] = None
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0]
        eval_obj = {}
        eval_obj['type'] = 'Miss'
        for metric in self.config['entity_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        return eval_obj
    def process_match(self, truth_entity, pred_entity, series_step, forecast_step):
        truth_entity['data'] = self.raw_data[series_step+self.config['input_window'], truth_entity['loc_ids'], 0]
        pred_entity['data'] = np.array([self.pred_series[series_step][k][forecast_step] for k in pred_entity['loc_ids']])
        eval_obj = {}
        for metric in self.config['entity_metrics']:
            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, pred_entity)
        eval_obj['type'] = self.judge_hits_misses(eval_obj)
        return eval_obj
        
    def compute_entities_errors(self):
        self.get_predict_entities()
        point_locs_array = np.array(self.geo_coords)
        # match the entities in predictions and ground truth
        # first compute the errors for entities in ground truth
        truth_entities_series = [[] for _ in range(len(self.all_pred_entities))]
        
        for truth_entities in self.step_entities:
            if len(truth_entities == 0): continue
            series_id = truth_entities[0]['step_id'] - self.config['input_window']
            truth_entities_series[series_id] = truth_entities
        for i, truth_entities in enumerate(truth_entities_series):
            step_pred_entities = self.all_pred_entities[i]
            for j, pred_entities in enumerate(step_pred_entities):
                if len(truth_entities) == 0 and len(pred_entities) == 0: continue
                elif len(truth_entities) == 0:
                    # 所有的pred_entities算为false alarm
                    for pred_entity in pred_entities:
                        eval_obj = self.process_false_alarm(pred_entity, i, j)
                        self.all_evaluated_pairs.append(eval_obj)
                elif len(pred_entities) == 0:
                    # 所有的truth_entities算为miss
                    for truth_entity in truth_entities:
                        eval_obj = self.process_miss(truth_entity, i, j)
                        self.all_evaluated_pairs.append(eval_obj)
                else:
                    # 需要对step内所有的pred_entity和truth_entity进行匹配
                    if len(truth_entities) == 1 and len(pred_entities) == 1:
                        truth_entity = truth_entities[0]
                        pred_entity = pred_entities[0]
                        # 直接看是否匹配
                        shift_dis = np.linalg.norm(truth_entity['centroid'] - pred_entity['centroid'])
                        truth_entity['data'] = self.raw_data[i+self.config['input_window'], truth_entity['loc_ids'], 0]
                        pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in pred_entity['loc_ids']])
                        if shift_dis < self.config['event_params']['related_dis_th']:
                            eval_obj = self.process_match(self, truth_entity, pred_entity, i, j)
                            self.all_evaluated_pairs.append(eval_obj)
                        else:
                            eval_obj_far = self.process_false_alarm(pred_entity, i, j)
                            eval_obj_miss = self.process_miss(truth_entity, i, j)
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
                        for k in range(len(pred_entity_matches)):

                            if pred_entity_matches[k] == 0:  # false alarm
                                eval_obj = self.process_false_alarm(pred_entities[k], i, j)
                                self.all_evaluated_pairs.append(eval_obj)
                            elif pred_entity_matches[k] == 1:
                                truth_entity_id = pred_entity_matches[k][0]
                                truth_entity = truth_entities[truth_entity_id]
                                pred_entity_ids = truth_entity_matches[truth_entity_id]
                                # 合并pred entities
                                agg_pred_entity = {}
                                agg_pred_entity['loc_ids'] = []
                                for pred_entity_id in pred_entity_ids:
                                    agg_pred_entity['loc_ids'] += pred_entities[pred_entity_id]['loc_ids']
                                agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                agg_pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in agg_pred_entity['loc_ids']])
                                agg_pred_entity['intensity'] = np.mean(agg_pred_entity['data'])
                                grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                                eval_obj = self.process_match(truth_entity, agg_pred_entity, i, j)
                                self.all_evaluated_pairs.append(eval_obj)
                            elif pred_entity_matches[k] > 1:
                                # 先合并truth entities
                                truth_entity_ids = pred_entity_matches[k]
                                pred_entity_ids = []
                                agg_truth_entity = {}
                                agg_truth_entity['loc_ids'] = []
                                for truth_entity_id in truth_entity_ids:
                                    agg_truth_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                    pred_entity_ids += truth_entity_matches[truth_entity_id]
                                agg_truth_entity['area'] = len(agg_truth_entity['loc_ids'])
                                agg_truth_entity['data'] = self.raw_data[i+self.config['input_window'], agg_truth_entity['loc_ids'], 0]
                                agg_truth_entity['intensity'] = np.mean(agg_truth_entity['data'])
                                grid_weights = agg_truth_entity['data'] / np.sum(agg_truth_entity['data'])
                                focus_locs = point_locs_array[agg_truth_entity['loc_ids']]
                                agg_truth_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                                # 再合并相应的pred entities
                                pred_entity_ids = list(set(pred_entity_ids))
                                agg_pred_entity = {}
                                agg_pred_entity['loc_ids'] = []
                                for truth_entity_id in truth_entity_ids:
                                    agg_pred_entity['loc_ids'] += truth_entities[truth_entity_id]['loc_ids']
                                    pred_entity_ids += truth_entity_matches[truth_entity_id]
                                agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                                agg_pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in agg_pred_entity['loc_ids']])
                                agg_pred_entity['intensity'] = np.mean(agg_pred_entity['data'])
                                grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                                focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                                agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                                eval_obj = self.process_match(truth_entity, agg_pred_entity, i, j)
                                self.all_evaluated_pairs.append(eval_obj)
                        
                        
                    
                    elif len(truth_entities) == 1 and len(pred_entities) > 1:
                        truth_centroids = [entity['centroid'] for entity in truth_entities]
                        pred_centroids = [entity['centroid'] for entity in pred_entities]
                        distances = distance.cdist(truth_centroids, pred_centroids)
                        union_ids = []
                        un_union_ids = []
                        for k in range(distances.shape[1]):
                            if distances[0][k] < self.config['event_params']['related_dis_th']:
                                union_ids.append(k)
                            else:
                                un_union_ids.append(k)
                        agg_pred_entity = {}
                        agg_pred_entity['loc_ids'] = []
                        for union_id in union_ids:
                            agg_pred_entity['loc_ids'] += pred_entities[union_id]['loc_ids']
                        agg_pred_entity['area'] = len(agg_pred_entity['loc_ids'])
                        agg_pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in agg_pred_entity['loc_ids']])
                        agg_pred_entity['intensity'] = np.mean(agg_pred_entity['data'])
                        grid_weights = agg_pred_entity['data'] / np.sum(agg_pred_entity['data'])
                        focus_locs = point_locs_array[agg_pred_entity['loc_ids']]
                        agg_pred_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                        truth_entity = truth_entities[0]
                        truth_entity['data'] = self.raw_data[i+self.config['input_window'], truth_entity['loc_ids'], 0]
                        eval_obj = {}
                        for metric in self.config['entity_metrics']:
                            eval_obj[metric] = self.call_method_by_name(metric, truth_entity, agg_pred_entity)
                        eval_obj['type'] = self.judge_hits_misses(eval_obj)
                        self.all_evaluated_pairs.append(eval_obj)
                        for un_union_id in un_union_ids:
                            pred_entity = pred_entities[un_union_id]
                            pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in pred_entity['loc_ids']])
                            eval_obj = self.process_false_alarm(pred_entity, i, j)
                            self.all_evaluated_pairs.append(eval_obj)
                    elif len(truth_entities) > 1 and len(pred_entities) == 1:
                        truth_centroids = [entity['centroid'] for entity in truth_entities]
                        pred_centroids = [entity['centroid'] for entity in pred_entities]
                        distances = distance.cdist(truth_centroids, pred_centroids)
                        union_ids = []
                        un_union_ids = []
                        for k in range(distances.shape[0]):
                            if distances[0][k] < self.config['event_params']['related_dis_th']:
                                union_ids.append(k)
                            else:
                                un_union_ids.append(k)
                        agg_truth_entity = {}
                        agg_truth_entity['loc_ids'] = []
                        for union_id in union_ids:
                            agg_truth_entity['loc_ids'] += truth_entities[union_id]['loc_ids']
                        agg_truth_entity['area'] = len(agg_truth_entity['loc_ids'])
                        agg_truth_entity['data'] = self.raw_data[i+self.config['input_window'], agg_truth_entity['loc_ids'], 0]
                        agg_truth_entity['intensity'] = np.mean(agg_truth_entity['data'])
                        grid_weights = agg_truth_entity['data'] / np.sum(agg_truth_entity['data'])
                        focus_locs = point_locs_array[agg_truth_entity['loc_ids']]
                        agg_truth_entity['centroid'] = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                        pred_entity = pred_entities[0]
                        pred_entity['data'] = np.array([self.pred_series[i][k][j] for k in pred_entity['loc_ids']])
                        eval_obj = {}
                        for metric in self.config['entity_metrics']:
                            eval_obj[metric] = self.call_method_by_name(metric, agg_truth_entity, pred_entity)
                        eval_obj['type'] = self.judge_hits_misses(eval_obj)
                        self.all_evaluated_pairs.append(eval_obj)
                        for un_union_id in un_union_ids:
                            truth_entity = truth_entities[un_union_id]
                            truth_entity['data'] = self.raw_data[i+self.config['input_window'], truth_entity['loc_ids'], 0]
                            eval_obj = self.process_false_alarm(truth_entity, i, j)
                            self.all_evaluated_pairs.append(eval_obj)
                    elif len(truth_entities) > 1 and len(pred_entities) > 1:

    
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
            return ['Approx. correct', pred_entity['area']-truth_entity['area']]
        elif truth_entity['intensity'] - pred_entity['intensity'] > truth_entity['intensity'] * val_change_th:
            return ['Underestimate', pred_entity['area']-truth_entity['area']]
        elif pred_entity['intensity'] - truth_entity['intensity'] > truth_entity['intensity'] * val_change_th:
            return ['Overestimate', pred_entity['intensity']-truth_entity['intensity']]
      
    def shift_error(self, truth_entity, pred_entity):
        if truth_entity['centroid'] is None or pred_entity['centroid'] is None:
            return None
        else:
            return np.linalg.norm(truth_entity['centroid'] - pred_entity['centroid'])

    def union_points_error(self, truth_entity, pred_entity):
        return loss.masked_mae_np(truth_entity['data'], pred_entity['data'])
    
    def judge_hits_misses(self, eval_obj):
        return
    
    def events_clustering(self):
        return 