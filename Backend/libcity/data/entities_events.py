import os
import torch
import math
import numpy as np
import pandas as pd
import random
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
import hashlib

class Entities_Events():
    def __init__(self, config, raw_data, phases, geo_coords, event_params):
        self.config = config
        self.raw_data = raw_data
        self.phases = phases
        self.geo_coords = geo_coords
        self.event_params = event_params
        self.focus_th = self.config.get('focus_th', 115)
        self.event_types = ["Forming", "Merging", "Continuing", "Growing", "Shape Changing", "Shrinking", "Splitting", "Dissolving"]
        self.all_entities = []
        self.step_entities = []
        self.all_events = []
    
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
        related_dis_th = float(self.event_params['related_dis_th'])
        loc_ids_1 = entity1['loc_ids']
        loc_ids_2 = entity2['loc_ids']
        if set(loc_ids_1).intersection(set(loc_ids_2)): return 1
        for loc1 in loc_ids_1:
            for loc2 in loc_ids_2:
                cur_dis = self.haversine_distance(self.geo_coords[loc1], self.geo_coords[loc2])
                if cur_dis < related_dis_th:
                    return 1
        return 0
        
    
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
    
    def get_all_entities_events(self):
        point_locs_array = np.array(self.geo_coords)
        min_grid_num = int(self.event_params['min_grid_num'])
        connected_dis_th = float(self.event_params['connected_dis_th'])
        move_dis_th = float(self.event_params['move_dis_th'])
        related_dis_th = float(self.event_params['related_dis_th'])
        val_change_th = float(self.event_params['val_change_th'])
        area_change_th = float(self.event_params['area_change_th'])
        area_consist_th = float(self.event_params['area_remain_th'])
        
        for phase_id, phase in enumerate(self.phases):
            focus_entitys = []
            phase_data = self.raw_data[:,:,0][phase['start']:(phase['end']+1)]
            step_num, loc_num = phase_data.shape
            phase_focus_flag = phase_data >= self.focus_th
            for i in range(step_num):
                step_focus_num = np.count_nonzero(phase_focus_flag[i])
                step_focus_entity = []
                if step_focus_num == 0:
                    focus_entitys.append(step_focus_entity)
                    self.step_entities.append(step_focus_entity)
                    continue
                step_focus_loc_ids = np.where(phase_focus_flag[i])[0].astype(int)
                step_focus_locs = point_locs_array[phase_focus_flag[i]]
                # identify focus clusters
                dis_mtx = haversine_distances(np.radians(step_focus_locs)) * 6371000 / 1000
                step_entities = self.get_connected_components(dis_mtx, connected_dis_th, min_grid_num)
                
                for j, entity in enumerate(step_entities):
                    entity_obj = {}
                    # print(cur_ids_in_entity)
                    entity_obj['step_id'] = phase['start'] + i
                    entity_obj['loc_ids'] = step_focus_loc_ids[entity].tolist()
                    entity_obj['id'] = f"{entity_obj['step_id']}-{str(j)}"
                    entity_obj['area'] = len(entity_obj['loc_ids'])
                    entity_focus_data = phase_data[i, entity_obj['loc_ids']]
                    entity_obj['max_val'] = float(np.max(entity_focus_data))
                    entity_obj['intensity'] = float(np.mean(entity_focus_data))
                    grid_weights = entity_focus_data / np.sum(entity_focus_data)
                    focus_locs = point_locs_array[entity_obj['loc_ids']]
                    step_centroid = np.sum(focus_locs * grid_weights[:, np.newaxis], axis=0)
                    entity_obj['centroid'] = step_centroid.tolist()
                    entity_obj['pre_match'] = 0
                    entity_obj['post_match'] = 0
                    entity_obj['post_match_ids'] = []      
                    entity_obj['pre_match_ids'] = []
                    step_focus_entity.append(entity_obj)
                self.step_entities.append(step_focus_entity)
                focus_entitys.append(step_focus_entity)
            phase['focus_entities'] = focus_entitys
            self.all_entities.append(focus_entitys)
            prev_entities = []
            phase_events = []
            phase_event_objs = []
            for i, entities in enumerate(focus_entitys):
                if i == 0 or len(focus_entitys[i-1]) == 0:
                    for j, entity in enumerate(entities):
                        # 均为forming事件
                        area_obj, val_obj, move_obj = {}, {}, {}
                        area_obj['type'] = 'Forming'
                        area_obj['change_area'] = entity['area']
                        val_obj['type'] = 'Increasing'
                        pre_intensity = np.mean(self.raw_data[i+phase['start']-1,entity['loc_ids'], 0])
                        val_obj['change_intensity'] = entity['intensity'] - pre_intensity
                        move_obj['type'] = 'Staying'
                        move_obj['move_dis'] = 0
                        event_obj = {
                            'phase_id': phase_id,
                            'step_id': i,
                            'area_type': area_obj['type'],
                            'change_area': area_obj['change_area'],
                            'pre_area': 0,
                            'intensity_type': val_obj['type'],
                            'pre_intensity': pre_intensity,
                            'change_intensity': val_obj['change_intensity'],
                            'move_type': move_obj['type'],
                            'move_dis': move_obj['move_dis']
                        }
                        phase_event_objs.append(event_obj)
                        phase_events.append((phase_id, i, None, j, area_obj, val_obj, move_obj))
                        self.all_events.append((phase_id, i, None, j, area_obj, val_obj, move_obj))
                elif len(focus_entitys[i]) == 0:
                    # 均为dissolving事件
                    for j, prev_entity in enumerate(focus_entitys[i-1]):
                        area_obj, val_obj, move_obj = {}, {}, {}
                        area_obj['type'] = 'Dissolving'
                        area_obj['change_area'] = prev_entity['area']
                        val_obj['type'] = 'Decreasing'
                        cur_intensity = np.mean(self.raw_data[i+phase['start'],prev_entity['loc_ids'], 0])
                        val_obj['change_intensity'] = cur_intensity - prev_entity['intensity']
                        move_obj['type'] = 'Staying'
                        move_obj['move_dis'] = 0
                        event_obj = {
                            'phase_id': phase_id,
                            'step_id': i,
                            'area_type': area_obj['type'],
                            'change_area': area_obj['change_area'],
                            'pre_area': prev_entity['area'],
                            'intensity_type': val_obj['type'],
                            'pre_intensity': prev_entity['intensity'],
                            'change_intensity': val_obj['change_intensity'],
                            'move_type': move_obj['type'],
                            'move_dis': move_obj['move_dis']
                        }
                        phase_event_objs.append(event_obj)
                        phase_events.append((phase_id, i, j, None, area_obj, val_obj, move_obj))
                        self.all_events.append((phase_id, i, j, None, area_obj, val_obj, move_obj))
                else:
                    prev_entities = focus_entitys[i-1]
                    entity_rel_mtx = np.zeros((len(prev_entities), len(entities)))
                    for j, prev_entity in enumerate(prev_entities):
                        for k, entity in enumerate(entities):
                            entity_rel_mtx[j][k] = self.compute_entity_match(prev_entity, entity)
                    
                    # 计算prev_entity和entity的match信息
                    for prev in range(entity_rel_mtx.shape[0]):
                        match_num = int(np.sum(entity_rel_mtx[prev]))
                        if match_num >= 0:
                            prev_entities[prev]['post_match'] = match_num
                            prev_entities[prev]['post_match_ids'] = np.where(entity_rel_mtx[prev] > 0)[0].tolist()
                    for cur in range(entity_rel_mtx.shape[1]):
                        match_num = int(np.sum(entity_rel_mtx[:, cur]))
                        if match_num >= 0:
                            entities[cur]['pre_match'] = match_num
                            entities[cur]['pre_match_ids'] = np.where(entity_rel_mtx[:,cur] > 0)[0].tolist()
                    # 先从prev entities判断是否存在dissolving、splitting
                    for prev, prev_entity in enumerate(prev_entities):
                        if prev_entity['post_match'] == 0:
                            # 是dissolving事件
                            area_obj, val_obj, move_obj = {}, {}, {}
                            area_obj['type'] = 'Dissolving'
                            area_obj['change_area'] = prev_entity['area']
                            val_obj['type'] = 'Decreasing'
                            cur_intensity = np.mean(self.raw_data[i+phase['start'],prev_entity['loc_ids'], 0])
                            val_obj['change_intensity'] = cur_intensity - prev_entity['intensity']
                            move_obj['type'] = 'Staying'
                            move_obj['move_dis'] = 0
                            event_obj = {
                                'phase_id': phase_id,
                                'step_id': i,
                                'area_type': area_obj['type'],
                                'change_area': area_obj['change_area'],
                                'pre_area': prev_entity['area'],
                                'intensity_type': val_obj['type'],
                                'pre_intensity': prev_entity['intensity'],
                                'change_intensity': val_obj['change_intensity'],
                                'move_type': move_obj['type'],
                                'move_dis': move_obj['move_dis']
                            }
                            phase_event_objs.append(event_obj)
                            phase_events.append((phase_id, i, prev, None, area_obj, val_obj, move_obj))
                            self.all_events.append((phase_id, i, prev, None, area_obj, val_obj, move_obj))
                        if prev_entity['post_match'] > 1:
                            # 是splitting事件
                            area_obj, val_obj, move_obj = {}, {}, {}
                            area_obj['type'] = 'Splitting'
                            rel_ids = prev_entity['post_match_ids']
                            area_obj['change_area'] = int(np.sum([entities[id]['area'] for id in rel_ids]) - prev_entity['area'])
                            # 计算intensity属性
                            rel_intensity = np.mean([entities[id]['intensity'] for id in rel_ids])
                            if rel_intensity - prev_entity['intensity'] > prev_entity['intensity'] * val_change_th:
                                val_obj['type'] = 'Increasing'
                            elif rel_intensity - prev_entity['intensity'] < -prev_entity['intensity'] * val_change_th:
                                val_obj['type'] = 'Decreasing'
                            else:
                                val_obj['type'] = 'Stable'
                            val_obj['change_intensity'] = rel_intensity - prev_entity['intensity']
                            # 计算move属性
                            rel_centroids = [entities[id]['centroid'] for id in rel_ids]
                            move_dis = np.mean([self.haversine_distance(prev_entity['centroid'], cur_cen) for cur_cen in rel_centroids])
                            if move_dis > move_dis_th:
                                move_obj['type'] = 'Moving'
                            else:
                                move_obj['type'] = 'Staying'
                            move_obj['move_dis'] = move_dis
                            event_obj = {
                                'phase_id': phase_id,
                                'step_id': i,
                                'area_type': area_obj['type'],
                                'change_area': area_obj['change_area'],
                                'pre_area': prev_entity['area'],
                                'intensity_type': val_obj['type'],
                                'pre_intensity': prev_entity['intensity'],
                                'change_intensity': val_obj['change_intensity'],
                                'move_type': move_obj['type'],
                                'move_dis': move_obj['move_dis']
                            }
                            phase_event_objs.append(event_obj)
                            phase_events.append((phase_id, i, prev, prev_entity['post_match_ids'], area_obj, val_obj, move_obj))
                            self.all_events.append((phase_id, i, prev, prev_entity['post_match_ids'], area_obj, val_obj, move_obj))
                    for cur, entity in enumerate(entities):
                        if entity['pre_match'] == 0:
                            # 为forming事件
                            area_obj, val_obj, move_obj = {}, {}, {}
                            area_obj['type'] = 'Forming'
                            area_obj['change_area'] = entity['area']
                            val_obj['type'] = 'Increasing'
                            pre_intensity = np.mean(self.raw_data[i+phase['start']-1,entity['loc_ids'], 0])
                            val_obj['change_intensity'] = entity['intensity'] - pre_intensity
                            move_obj['type'] = 'Staying'
                            move_obj['move_dis'] = 0
                            event_obj = {
                                'phase_id': phase_id,
                                'step_id': i,
                                'area_type': area_obj['type'],
                                'change_area': area_obj['change_area'],
                                'pre_area': 0,
                                'intensity_type': val_obj['type'],
                                'pre_intensity': pre_intensity,
                                'change_intensity': val_obj['change_intensity'],
                                'move_type': move_obj['type'],
                                'move_dis': move_obj['move_dis']
                            }
                            phase_event_objs.append(event_obj)
                            phase_events.append((phase_id, i, None, cur, area_obj, val_obj, move_obj))
                            self.all_events.append((phase_id, i, None, cur, area_obj, val_obj, move_obj))
                        if entity['pre_match'] > 1:
                            # 为merging事件
                            area_obj, val_obj, move_obj = {}, {}, {}
                            area_obj['type'] = 'Merging'
                            rel_ids = entity['pre_match_ids']
                            area_obj['change_area'] = int(entity['area'] - np.sum([prev_entities[id]['area'] for id in rel_ids]))
                            # 计算intensity属性
                            rel_intensity = np.mean([prev_entities[id]['intensity'] for id in rel_ids])
                            if entity['intensity'] - rel_intensity > rel_intensity * val_change_th:
                                val_obj['type'] = 'Increasing'
                            elif entity['intensity'] - rel_intensity < -rel_intensity * val_change_th:
                                val_obj['type'] = 'Decreasing'
                            else:
                                val_obj['type'] = 'Stable'
                            val_obj['change_intensity'] = entity['intensity'] - rel_intensity
                            # 计算move属性
                            rel_centroids = [prev_entities[id]['centroid'] for id in rel_ids]
                            move_dis = np.mean([self.haversine_distance(entity['centroid'], prev_cen) for prev_cen in rel_centroids])
                            if move_dis > move_dis_th:
                                move_obj['type'] = 'Moving'
                            else:
                                move_obj['type'] = 'Staying'
                            move_obj['move_dis'] = move_dis
                            event_obj = {
                                'phase_id': phase_id,
                                'step_id': i,
                                'area_type': area_obj['type'],
                                'change_area': area_obj['change_area'],
                                'pre_area': int(np.sum([prev_entities[id]['area'] for id in rel_ids])),
                                'intensity_type': val_obj['type'],
                                'pre_intensity': rel_intensity,
                                'change_intensity': val_obj['change_intensity'],
                                'move_type': move_obj['type'],
                                'move_dis': move_obj['move_dis']
                            }
                            phase_event_objs.append(event_obj)
                            phase_events.append((phase_id, i, entity['pre_match_ids'], cur, area_obj, val_obj, move_obj))
                            self.all_events.append((phase_id, i, entity['pre_match_ids'], cur, area_obj, val_obj, move_obj))
                        # 如何判断Continuing、Growing、Shrinking：这些都是一对一的关系。
                        # 判断每个只跟一个对应的entity的prev_entity是否也为1
                        if entity['pre_match'] == 1 and prev_entities[entity['pre_match_ids'][0]]['post_match'] == 1:
                            area_obj, val_obj, move_obj = {}, {}, {}
                            intersect_ids = np.intersect1d(prev_entity['loc_ids'], entity['loc_ids'])
                            union_ids = list(sorted(set(prev_entity['loc_ids']).union(set(entity['loc_ids']))))
                            area_obj['change_area'] = entity['area'] - prev_entity['area']
                            if area_obj['change_area'] > prev_entity['area'] * area_change_th:
                                area_obj['type'] = 'Growing'
                            elif area_obj['change_area'] < -prev_entity['area'] * area_change_th:
                                area_obj['type'] = 'Shrinking'
                            elif (intersect_ids.size / len(union_ids) >= area_consist_th):
                                area_obj['type'] = "Continuing"
                            else:
                                area_obj['type'] = "Shape Changing"
                            val_obj['change_intensity'] = entity['intensity'] - prev_entity['intensity']
                            if val_obj['change_intensity'] > prev_entity['intensity'] * val_change_th:
                                val_obj['type'] = 'Increasing'
                            elif val_obj['change_intensity'] < -prev_entity['intensity'] * val_change_th:
                                val_obj['type'] = 'Decreasing'
                            else:
                                val_obj['type'] = 'Stable'
                            move_obj['move_dis'] = self.haversine_distance(prev_entity['centroid'], entity['centroid'])
                            if move_obj['move_dis'] > move_dis_th:
                                move_obj['type'] = 'Moving'
                            else:
                                move_obj['type'] = 'Staying'
                            event_obj = {
                                'phase_id': phase_id,
                                'step_id': i,
                                'area_type': area_obj['type'],
                                'change_area': area_obj['change_area'],
                                'pre_area': prev_entity['area'],
                                'intensity_type': val_obj['type'],
                                'pre_intensity': prev_entity['intensity'],
                                'change_intensity': val_obj['change_intensity'],
                                'move_type': move_obj['type'],
                                'move_dis': move_obj['move_dis']
                            }
                            phase_event_objs.append(event_obj)
                            phase_events.append((phase_id, i, entity['pre_match_ids'][0], cur, area_obj, val_obj, move_obj))
                            self.all_events.append((phase_id, i, entity['pre_match_ids'][0], cur, area_obj, val_obj, move_obj))
            phase['phase_events'] = phase_events
            phase['phase_event_objs'] = phase_event_objs
        return self.phases
    
    def save_entities_events(self):
        params_str = '_'.join([str(value) for value in self.event_params.values() if not isinstance(value, list)])
        attr_str = '_'.join(sorted(self.event_params['attributes']))
        metric_str = '_'.join(sorted(self.event_params['error_metrics']))
        attributes_str = params_str + '_' + attr_str + '_' + metric_str
        file_hash = hashlib.md5(attributes_str.encode()).hexdigest()
        file_label = f"{file_hash[:8]}"
        
        entities_events_file = f"./entities_events/{self.config['dataset']}_{float(self.config['focus_th'])}_{file_label}.npz"
        if not os.path.exists(entities_events_file):
            np.savez(entities_events_file, entities=self.all_entities, events=self.all_events, allow_pickle=True)