import sys
sys.path.append('./')
from libcity.utils import ensure_dir
from collections import OrderedDict
import torch
import pandas as pd

import datetime
import numpy as np
import torch
import time
import math
import os
import copy
import json
from collections import Counter
# from main import compute_error_indicators, compute_failure_indicators, compute_focus_indicators
from scipy.stats import entropy  
# from seg_range_focus import seg_range_divergence, get_focus_cnt
from libcity.model import loss
import pyfpgrowth
import itertools
from functools import reduce
# from community import community_louvain 
import community
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial.distance import jensenshannon


# 遍历树，提取节点的频繁模式，形成子集
def is_subset(list1, list2):  
    set1 = set(list1)  
    set2 = set(list2)  
    return set1.issubset(set2)

def extract_frequent_subset_fpgrowth(data_infors, residuals, attr_data, attr_objs, fre_sup_th):
    all_attrs_strs = []
    # 判断每个事件的各个属性的str，并且形成str序列
    for i, data_infor in enumerate(data_infors):
        attrs_strs = []
        for attr in attr_objs.keys():
            for val in attr_objs[attr]['bins']:
                if val[0] <= data_infor[attr] < val[1]:
                    attrs_strs.append(attr + '=' + str(val))
                    break
        all_attrs_strs.append(attrs_strs)
    # print(event_attrs_strs)
    sup_th_num = round(len(data_infors) * fre_sup_th)
    patterns = pyfpgrowth.find_frequent_patterns(all_attrs_strs, support_threshold=sup_th_num)
    patterns_list = list(patterns.keys())
    pattern_objs = {}
    for j, pattern in enumerate(patterns_list):
        pattern_objs[pattern] = {
            'indices': []
        }
    # converted_patterns = {str(key): value for key, value in patterns.items()}
    for i, attrs_strs in enumerate(all_attrs_strs):
        for j, pattern in enumerate(patterns_list):
            if is_subset(pattern, attrs_strs):
            # if all(item in attrs_strs for item in pattern):
                pattern_objs[pattern]['indices'].append(i)
    pattern_objs_list = []
    # subset_id = 0
    for pattern in pattern_objs.keys():
        pattern_indices = pattern_objs[pattern]['indices']
        pos_residuals = residuals[pattern_indices][residuals[pattern_indices] >= 0]
        neg_residuals = residuals[pattern_indices][residuals[pattern_indices] < 0]
        pattern_objs[pattern]['sup_num'] = len(pattern_indices)        
        pattern_objs[pattern]['pos_res_num'] = int(pos_residuals.size)
        pattern_objs[pattern]['neg_res_num'] = int(neg_residuals.size)
        pattern_objs[pattern]['sup_rate'] = round(len(pattern_indices) / len(data_infors), 4)
        pattern_objs[pattern]['residual_abs'] = round(np.mean(np.abs(residuals[pattern_indices])), 4)
        if pos_residuals.size == 0: pattern_objs[pattern]['residual_pos'] = 0
        else: pattern_objs[pattern]['residual_pos'] = np.mean(pos_residuals)
        if neg_residuals.size == 0: pattern_objs[pattern]['residual_neg'] = 0
        else: pattern_objs[pattern]['residual_neg'] = np.mean(neg_residuals)
        pattern_objs[pattern]['residual_std'] = np.std(residuals[pattern_indices])
        pattern_objs[pattern]['residual_entropy'] = entropy(np.abs(residuals[pattern_indices]))
        pattern_objs[pattern]['subset_attrs'] = list(pattern)
        # pattern_objs[pattern]['subset_id'] = subset_id
        # subset_id += 1
        pattern_objs_list.append(pattern_objs[pattern])
    
    sorted_subsets = sorted(pattern_objs_list, key=lambda x: x['residual_abs'], reverse=True)
    for i, subset in enumerate(sorted_subsets):
        subset['subset_id'] = i
    # print(sorted_subsets)
    
    return sorted_subsets

def merge_dicts(lst):
    index_dict = {}
    for d in lst:
        index_list = tuple(d['indices'])
        if index_list not in index_dict:
            index_dict[index_list] = set(d['subset_attrs'])
        else:
            index_dict[index_list].update(d['subset_attrs'])

    result = []
    for index_list, string_set in index_dict.items():
        result.append({'indices': list(index_list), 'string_array': list(string_set)})

    return result

def slicing_with_polarity_pruning(range_infor_all, residuals, attr_data, attr_objs, forecast_scopes, val_bins, fre_sup_th, err_diff_th, err_th, redundancy_th, purity):
    print('start to run function')
    subset_infor = {}
    # 为每个range计算数据的索引，存储为dict
    range_infor_strs = {}
    range_data_ids = {}
    all_attrs = list(range_infor_all.keys())
    for attr in range_infor_all:
        range_infor_strs[attr] = []
        for range_infor in range_infor_all[attr]:
            cur_range = range_infor['range']
            cur_range_ids = (attr_data[attr] >= cur_range[0]) & (attr_data[attr] <= cur_range[1])
            range_infor_strs[attr].append(range_infor['range_str'])
            range_data_ids[range_infor['range_str']] = cur_range_ids
    
    all_subsets = []
    mean_residual = np.mean(np.abs(residuals))
    print(mean_residual)
    sup_th_num = round(residuals.size * fre_sup_th)
    valid_attrs = {}
    all_valid_subsets_dict = {}
    all_valid_subsets_dict[1] = {}
    subset_indices_dict = {}
    
    for attr in range_infor_all:
        valid_attrs[attr] = []
        cur_ranges = range_infor_strs[attr]
        for cur_range in cur_ranges:
            cur_subset_infor = {}
            cur_subset_infor['indices'] = range_data_ids[cur_range]
            cur_subset_error = residuals[cur_subset_infor['indices']]
            cur_subset_infor['err_polarity'] = None
            if np.mean(cur_subset_error) >= 0: cur_subset_infor['err_polarity'] = 'pos'
            else: cur_subset_infor['err_polarity'] = 'neg'
            cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
            if purity == 'High_Error' and cur_subset_infor['residual_abs'] < err_th:
                continue
            if purity == 'Low_Error' and cur_subset_infor['residual_abs'] > err_th:
                continue
            cur_subset_infor['subset_attrs'] = [cur_range]
            cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
            cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
            subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
            subset_neg_errs = cur_subset_error[cur_subset_error < 0]
            cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size)
            cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
            if subset_pos_errs.size == 0: cur_subset_infor['residual_pos'] = 0
            else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
            if subset_neg_errs.size == 0: cur_subset_infor['residual_neg'] = 0
            else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
            cur_subset_infor['residual_std'] = np.std(cur_subset_error)
            # cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
            all_valid_subsets_dict[1][frozenset([cur_range])] = cur_subset_infor
            valid_attrs[attr].append(cur_range)
            all_subsets.append(cur_subset_infor)
    # print('all_valid_subsets_dict[1]', all_valid_subsets_dict[1])
    all_valid_subsets_dict[2] = {}
    # 存储所有组合的列表
    all_combinations = set()
    
    # for key1, key2 in itertools.product(range_infor_strs.keys(), repeat=2):
    #     # 如果两个键相同，跳过
    #     if key1 == key2: continue
    #     # 对应的值数组进行两两组合
    #     for value1, value2 in itertools.product(range_infor_strs[key1], range_infor_strs[key2]):
    #         all_combinations.add(tuple(sorted((value1, value2))))
    for key1, key2 in itertools.product(valid_attrs.keys(), repeat=2):
        # 如果两个键相同，跳过
        if key1 == key2: continue
        # 对应的值数组进行两两组合
        for value1, value2 in itertools.product(valid_attrs[key1], valid_attrs[key2]):
            all_combinations.add(tuple(sorted((value1, value2))))
    # print('all_combinations', all_combinations)
    for cur_subset in all_combinations:
        cur_subset_infor = {}
        # print('cur_subset', frozenset([cur_subset[0]]), frozenset([cur_subset[1]]))
        cur_subset_infor['indices'] = all_valid_subsets_dict[1][frozenset([cur_subset[0]])]['indices'] & all_valid_subsets_dict[1][frozenset([cur_subset[1]])]['indices']
        cur_subset_error = residuals[cur_subset_infor['indices']]
        cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
        if np.mean(cur_subset_error) >= 0: cur_subset_infor['err_polarity'] = 'pos'
        else: cur_subset_infor['err_polarity'] = 'neg'
        
        if purity == 'High_Error' and cur_subset_infor['residual_abs'] < err_th:
            continue
        if purity == 'Low_Error' and cur_subset_infor['residual_abs'] > err_th:
            continue
        cur_subset_infor['subset_attrs'] = list(cur_subset)
        cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
        if (cur_subset_infor['sup_num'] < sup_th_num): continue
        cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
        # subset_ids_flat = np.where(subset_ids == 1)[0]
        # subset_multi_ids = np.unravel_index(subset_ids_flat, (n_sample, window_size, n_loc))
        subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
        subset_neg_errs = cur_subset_error[cur_subset_error < 0]
        base_subset_1 = all_valid_subsets_dict[1][frozenset([cur_subset[0]])]
        base_subset_2 = all_valid_subsets_dict[1][frozenset([cur_subset[1]])]
        if abs(cur_subset_infor['residual_abs'] - base_subset_1['residual_abs']) < err_diff_th or abs(cur_subset_infor['residual_abs'] - base_subset_2['residual_abs']) < err_diff_th: continue
        cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size)
        cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
        if cur_subset_infor['pos_res_num'] == 0: cur_subset_infor['residual_pos'] = 0
        else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
        if cur_subset_infor['neg_res_num'] == 0: cur_subset_infor['residual_neg'] = 0
        else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
        cur_subset_infor['residual_std'] = np.std(cur_subset_error)
        # cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
        all_valid_subsets_dict[2][frozenset(cur_subset)] = cur_subset_infor
        all_subsets.append(cur_subset_infor)
    valid_ranges_two = list(all_valid_subsets_dict[2].keys())
    for r in range(3, len(list(range_infor_strs.keys()))+1):
        print(f"start {r} items computing")
        all_valid_subsets_dict[r] = {}
        # pre_subsets = [list(subset) for subset in list(all_valid_subsets_dict[r-1].keys())]
        pre_subsets = list(all_valid_subsets_dict[r-1].keys())
        # print(pre_subsets)
        for i in range(len(pre_subsets)):
            for j in range(i + 1, len(pre_subsets)):
                diff_ranges = frozenset(set(pre_subsets[i]) ^ set(pre_subsets[j]))
                if len(diff_ranges) == 2 and (diff_ranges in valid_ranges_two):
                    cur_subset = frozenset(set(pre_subsets[i]) | set(pre_subsets[j]))
                    if cur_subset in all_valid_subsets_dict[r]: continue
                    cur_subset_infor = {}
                    cur_subset_infor['indices'] = all_valid_subsets_dict[r-1][pre_subsets[i]]['indices'] & all_valid_subsets_dict[r-1][pre_subsets[j]]['indices']
                    cur_subset_error = residuals[cur_subset_infor['indices']]
                    cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
                    if np.mean(cur_subset_error) >= 0: cur_subset_infor['err_polarity'] = 'pos'
                    else: cur_subset_infor['err_polarity'] = 'neg'
                    if purity == 'High_Error' and cur_subset_infor['residual_abs'] < err_th:
                        continue
                    if purity == 'Low_Error' and cur_subset_infor['residual_abs'] > err_th:
                        continue
                    cur_subset_infor['subset_attrs'] = list(cur_subset)
                    cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
                    if (cur_subset_infor['sup_num'] < sup_th_num): continue
                    cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
                    # subset_ids_flat = np.where(subset_ids == 1)[0]
                    # subset_multi_ids = np.unravel_index(subset_ids_flat, (n_sample, window_size, n_loc))
                    subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
                    subset_neg_errs = cur_subset_error[cur_subset_error < 0]
                    base_subset_1 = all_valid_subsets_dict[r-1][pre_subsets[i]]
                    base_subset_2 = all_valid_subsets_dict[r-1][pre_subsets[j]]
                    if abs(cur_subset_infor['residual_abs'] - base_subset_1['residual_abs']) < err_diff_th or abs(cur_subset_infor['residual_abs'] - base_subset_2['residual_abs']) < err_diff_th: continue
                    cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size) 
                    cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
                    if cur_subset_infor['pos_res_num'] == 0: cur_subset_infor['residual_pos'] = 0
                    else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
                    if cur_subset_infor['neg_res_num'] == 0: cur_subset_infor['residual_neg'] = 0
                    else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
                    
                    cur_subset_infor['residual_std'] = np.std(cur_subset_error)
                    # cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
                    all_valid_subsets_dict[r][frozenset(cur_subset)] = cur_subset_infor
                    all_subsets.append(cur_subset_infor)
        
    for subset in all_subsets:
        subset['range_val'] = {}
        for j in range(len(subset['subset_attrs'])):
            cur_attr_str_split = subset['subset_attrs'][j].split('=')
            cur_attr_type = cur_attr_str_split[0]
            cur_attr_range = eval(cur_attr_str_split[1])
            subset['range_val'][cur_attr_type] = cur_attr_range
    sorted_valid_subsets = sorted(all_subsets, key=lambda x: x['residual_abs'], reverse=True)
    for i, subset in enumerate(sorted_valid_subsets):
        # 计算误差分布
        # cur_data_ids = subset['indices_list']
        cur_data_ids = subset['indices']
        cur_residual = residuals[cur_data_ids]
        
        residual_bins = val_bins['residual_bins']
        mid_bins = val_bins['mid_bins']
        pos_extreme_bins = val_bins['pos_extreme_bins']
        neg_extreme_bins = val_bins['neg_extreme_bins']
        
        cur_residual_hist, _ = np.histogram(cur_residual, bins=residual_bins, density=False)
        cur_mid_hist, _ = np.histogram(cur_residual, bins=mid_bins, density=False)
        cur_pos_extreme_hist, _ = np.histogram(cur_residual, bins=pos_extreme_bins, density=False)
        cur_neg_extreme_hist, _ = np.histogram(cur_residual, bins=neg_extreme_bins, density=False)
        subset['residual_hist'] = cur_residual_hist.tolist()
        subset['mid_hist'] = cur_mid_hist.tolist()
        subset['pos_extreme_hist'] = cur_pos_extreme_hist.tolist()
        subset['neg_extreme_hist'] = cur_neg_extreme_hist.tolist()
        
        cur_residual_hist_sums = np.histogram(cur_residual, bins=residual_bins, weights=cur_residual)[0]
        cur_pos_extreme_hist_sums = np.histogram(cur_residual, bins=pos_extreme_bins, weights=cur_residual)[0]
        cur_neg_extreme_hist_sums = np.histogram(cur_residual, bins=neg_extreme_bins, weights=cur_residual)[0]
        
        subset['residual_hist_mean'] = np.divide(cur_residual_hist_sums, cur_residual_hist, where=(cur_residual_hist > 0)).tolist()
        subset['pos_extreme_hist_mean'] = np.divide(cur_pos_extreme_hist_sums, cur_pos_extreme_hist, where=(cur_pos_extreme_hist > 0)).tolist()
        subset['neg_extreme_hist_mean'] = np.divide(cur_neg_extreme_hist_sums, cur_neg_extreme_hist, where=(cur_neg_extreme_hist > 0)).tolist()
        
        subset['residual_hist_normalize'] = np.round(cur_residual_hist / np.max(cur_residual_hist), 6).tolist()
        
        # 计算属性分布
        subset['subset_id'] = i
        subset['attrs_hist'] = {}
        for attr in all_attrs:
            # cur_attr_data = attr_data[attr][subset['indices_list']]
            cur_attr_data = attr_data[attr][subset['indices']]
            cur_attr_hist, _ = np.histogram(cur_attr_data, bins=attr_objs[attr]['bin_edges'], density=False)
            subset['attrs_hist'][attr] = cur_attr_hist.tolist()
    
    # 合并重叠度高的slice
    # jsd_th = 0.1
    # G = nx.Graph()
    # G.add_nodes_from(range(len(all_subsets)))
    # redundancy_mtx = np.zeros((len(all_subsets), len(all_subsets)))
    # zero_index = val_bins['residual_bins'].index(0)
    # for i in range(len(all_subsets)):
    #     for j in range(i+1, len(all_subsets)):
    #         # 基于jaccard相似性构建重叠度矩阵
    #         subset_join = sorted_valid_subsets[i]['indices'] & sorted_valid_subsets[j]['indices']
    #         subset_union = sorted_valid_subsets[i]['indices'] | sorted_valid_subsets[j]['indices']
    #         if np.count_nonzero(subset_join) == 0 or np.count_nonzero(subset_union) == 0: continue
    #         redundancy_mtx[i][j] = np.count_nonzero(subset_join) / np.count_nonzero(subset_union)
    #         redundancy_mtx[j][i] = redundancy_mtx[i][j]
    #         # 计算子集的误差分布之间的jensen-shannon散度
    #         pos_resi_hist_i = sorted_valid_subsets[i]['residual_hist_normalize'][zero_index:]
    #         pos_resi_hist_j = sorted_valid_subsets[j]['residual_hist_normalize'][zero_index:]
    #         neg_resi_hist_i = sorted_valid_subsets[i]['residual_hist_normalize'][:zero_index]
    #         neg_resi_hist_j = sorted_valid_subsets[j]['residual_hist_normalize'][:zero_index]
    #         jsd_pos = jensenshannon(pos_resi_hist_i, pos_resi_hist_j, base=2) ** 2
    #         jsd_neg = jensenshannon(neg_resi_hist_i, neg_resi_hist_j, base=2) ** 2
    #         if redundancy_mtx[i][j] > redundancy_th and jsd_pos < jsd_th and jsd_neg < jsd_th:
    #             G.add_edge(i, j, weight=redundancy_mtx[i][j])
    
    # union_subset_list = []
    # ids_in_community = []
    # if len(G) > 0 and G.number_of_edges() > 0:
    #     communities = list(greedy_modularity_communities(G))
    #     for i, community in enumerate(communities):
    #         cur_id_list = list(community)
    #         if len(cur_id_list) == 1: continue
    #         union_indices = np.zeros(sorted_valid_subsets[0]['indices'].shape, dtype=bool)
    #         union_subset = {}
    #         union_subset['contain_subsets'] = []
    #         union_subset['subset_attrs'] = []
    #         for subset_id in cur_id_list:
    #             sorted_valid_subsets[subset_id]['merged_subset'] = True
    #             ids_in_community.append(subset_id)
    #             union_subset['contain_subsets'].append(sorted_valid_subsets[subset_id])
    #             cur_indices = sorted_valid_subsets[subset_id]['indices']
    #             union_indices = union_indices | cur_indices
    #             union_subset['subset_attrs'] = union_subset['subset_attrs'] + sorted_valid_subsets[subset_id]['subset_attrs']
    #         union_subset['subset_attrs'] = list(set(union_subset['subset_attrs']))
    #         union_subset['range_val'] = {}
    #         for j in range(len(union_subset['subset_attrs'])):
    #             cur_attr_str_split = union_subset['subset_attrs'][j].split('=')
    #             cur_attr_type = cur_attr_str_split[0]
    #             cur_attr_range = eval(cur_attr_str_split[1])
    #             union_subset['range_val'][cur_attr_type] = cur_attr_range
    #         union_subset['indices'] = union_indices
    #         # union_subset['indices_list'] = np.where(union_subset['indices'] == True)[0].tolist()
    #         union_subset['residual_abs'] = round(np.mean(np.abs(residuals[union_indices])), 4)
    #         union_subset['sup_num'] = np.where(union_subset['indices'] == True)[0].size
    #         union_subset['pos_res_num'] = np.count_nonzero(residuals[union_indices] > 0)
    #         union_subset['neg_res_num'] = np.count_nonzero(residuals[union_indices] < 0)
    #         union_subset['sup_rate'] = round(union_subset['sup_num'] / residuals.size, 4)
    #         # 计算属性和误差的分布 - (slice的属性用文字还是hist，感觉可以都用！)
    #         # 计算误差分布
    #         cur_residual = residuals[union_indices]
    #         # cur_resi_hist, _ = np.histogram(cur_residual, bins=val_bins, density=False)
    #         # union_subset['residual_hist'] = cur_resi_hist.tolist()
    #         # union_subset['residual_hist_normalize'] = np.round(cur_resi_hist / np.max(cur_resi_hist), 6).tolist()
    #         residual_bins = val_bins['residual_bins']
    #         mid_bins = val_bins['mid_bins']
    #         pos_extreme_bins = val_bins['pos_extreme_bins']
    #         neg_extreme_bins = val_bins['neg_extreme_bins']
            
    #         cur_residual_hist, _ = np.histogram(cur_residual, bins=residual_bins, density=False)
    #         cur_mid_hist, _ = np.histogram(cur_residual, bins=mid_bins, density=False)
    #         cur_pos_extreme_hist, _ = np.histogram(cur_residual, bins=pos_extreme_bins, density=False)
    #         cur_neg_extreme_hist, _ = np.histogram(cur_residual, bins=neg_extreme_bins, density=False)
    #         union_subset['residual_hist'] = cur_residual_hist.tolist()
    #         union_subset['mid_hist'] = cur_mid_hist.tolist()
    #         union_subset['pos_extreme_hist'] = cur_pos_extreme_hist.tolist()
    #         union_subset['neg_extreme_hist'] = cur_neg_extreme_hist.tolist()
            
    #         cur_residual_hist_sums = np.histogram(cur_residual, bins=residual_bins, weights=cur_residual)[0]
    #         cur_pos_extreme_hist_sums = np.histogram(cur_residual, bins=pos_extreme_bins, weights=cur_residual)[0]
    #         cur_neg_extreme_hist_sums = np.histogram(cur_residual, bins=neg_extreme_bins, weights=cur_residual)[0]
            
    #         union_subset['residual_hist_mean'] = np.divide(cur_residual_hist_sums, cur_residual_hist, where=(cur_residual_hist > 0)).tolist()
    #         union_subset['pos_extreme_hist_mean'] = np.divide(cur_pos_extreme_hist_sums, cur_pos_extreme_hist, where=(cur_pos_extreme_hist > 0)).tolist()
    #         union_subset['neg_extreme_hist_mean'] = np.divide(cur_neg_extreme_hist_sums, cur_neg_extreme_hist, where=(cur_neg_extreme_hist > 0)).tolist()
            
    #         union_subset['residual_hist_normalize'] = np.round(cur_residual_hist / np.max(cur_residual_hist), 6).tolist()
            
    #         # 计算属性分布
    #         union_subset['subset_id'] = i
    #         union_subset['attrs_hist'] = {}
            
    #         for attr in all_attrs:
    #             cur_attr_data = attr_data[attr][union_indices]
    #             cur_attr_hist, _ = np.histogram(cur_attr_data, bins=attr_objs[attr]['bin_edges'], density=False)
    #             union_subset['attrs_hist'][attr] = cur_attr_hist.tolist()
    #         union_subset_list.append(union_subset)
    # # 从原subset中删除在community中的subset，并加入union subset
    
    # individual_subsets = [item for i, item in enumerate(sorted_valid_subsets) if i not in ids_in_community]
    # final_subsets = union_subset_list + individual_subsets
    # sorted_final_subsets = sorted(final_subsets, key=lambda x: x['residual_abs'], reverse=True)
    sorted_final_subsets = sorted_valid_subsets
    
    return sorted_final_subsets
    
def slicing_with_err_th(range_infor_all, residuals, attr_data, attr_objs, forecast_scopes, val_bins, fre_sup_th, err_diff_th, err_th, redundancy_th):
    print('start to run function')
    # residuals_abs_mean = residuals[]
    subset_infor = {}
    # 为每个range计算数据的索引，存储为dict
    range_infor_strs = {}
    range_data_ids = {}
    all_attrs = list(range_infor_all.keys())
    for attr in range_infor_all:
        range_infor_strs[attr] = []
        for range_infor in range_infor_all[attr]:
            cur_range = range_infor['range']
            cur_range_ids = (attr_data[attr] >= cur_range[0]) & (attr_data[attr] <= cur_range[1])
            range_infor_strs[attr].append(range_infor['range_str'])
            range_data_ids[range_infor['range_str']] = cur_range_ids
    # print('finish step 1')
    # 通过itertools来获取所有可能的range组合：
    # 1. 通过itertools获取所有可能的attr组合：
    # 逐步获得候选的子集
    all_subsets = []
    sup_th_num = round(residuals.size * fre_sup_th)
    all_valid_subsets_dict = {}
    all_valid_subsets_dict[1] = {}
    subset_indices_dict = {}
    
    for attr in range_infor_all:
        cur_ranges = range_infor_strs[attr]
        for cur_range in cur_ranges:
            cur_subset_infor = {}
            cur_subset_infor['subset_attrs'] = [cur_range]
            cur_subset_infor['indices'] = range_data_ids[cur_range]
            cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
            cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
            cur_subset_error = residuals[cur_subset_infor['indices']]
            subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
            subset_neg_errs = cur_subset_error[cur_subset_error < 0]
            cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
            cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size)
            cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
            if subset_pos_errs.size == 0: cur_subset_infor['residual_pos'] = 0
            else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
            if subset_neg_errs.size == 0: cur_subset_infor['residual_neg'] = 0
            else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
            cur_subset_infor['residual_std'] = np.std(cur_subset_error)
            cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
            all_valid_subsets_dict[1][frozenset([cur_range])] = cur_subset_infor
            all_subsets.append(cur_subset_infor)
    print('finish step 2')
    all_valid_subsets_dict[2] = {}
    # 存储所有组合的列表
    all_combinations = set()
    for key1, key2 in itertools.product(range_infor_strs.keys(), repeat=2):
        # 如果两个键相同，跳过
        if key1 == key2: continue
        # 对应的值数组进行两两组合
        for value1, value2 in itertools.product(range_infor_strs[key1], range_infor_strs[key2]):
            all_combinations.add(tuple(sorted((value1, value2))))
    
    for cur_subset in all_combinations:
        cur_subset_infor = {}
        cur_subset_infor['subset_attrs'] = list(cur_subset)
        cur_subset_infor['indices'] = all_valid_subsets_dict[1][frozenset([cur_subset[0]])]['indices'] & all_valid_subsets_dict[1][frozenset([cur_subset[1]])]['indices']
        cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
        if (cur_subset_infor['sup_num'] < sup_th_num): continue
        cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
        # subset_ids_flat = np.where(subset_ids == 1)[0]
        # subset_multi_ids = np.unravel_index(subset_ids_flat, (n_sample, window_size, n_loc))
        cur_subset_error = residuals[cur_subset_infor['indices']]
        subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
        subset_neg_errs = cur_subset_error[cur_subset_error < 0]
        cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
        if abs(cur_subset_infor['residual_abs'] - all_valid_subsets_dict[1][frozenset([cur_subset[0]])]['residual_abs']) < err_diff_th or abs(cur_subset_infor['residual_abs'] - all_valid_subsets_dict[1][frozenset([cur_subset[1]])]['residual_abs']) < err_diff_th: continue
        cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size)
        cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
        if cur_subset_infor['pos_res_num'] == 0: cur_subset_infor['residual_pos'] = 0
        else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
        if cur_subset_infor['neg_res_num'] == 0: cur_subset_infor['residual_neg'] = 0
        else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
        cur_subset_infor['residual_std'] = np.std(cur_subset_error)
        cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
        all_valid_subsets_dict[2][frozenset(cur_subset)] = cur_subset_infor
        all_subsets.append(cur_subset_infor)
    print('finish step 3')
    valid_ranges_two = list(all_valid_subsets_dict[2].keys())
    for r in range(3, len(list(range_infor_strs.keys()))+1):
        print(f"start {r} items computing")
        all_valid_subsets_dict[r] = {}
        # pre_subsets = [list(subset) for subset in list(all_valid_subsets_dict[r-1].keys())]
        pre_subsets = list(all_valid_subsets_dict[r-1].keys())
        # print(pre_subsets)
        for i in range(len(pre_subsets)):
            for j in range(i + 1, len(pre_subsets)):
                diff_ranges = frozenset(set(pre_subsets[i]) ^ set(pre_subsets[j]))
                if len(diff_ranges) == 2 and (diff_ranges in valid_ranges_two):
                    cur_subset = frozenset(set(pre_subsets[i]) | set(pre_subsets[j]))
                    if cur_subset in all_valid_subsets_dict[r]: continue
                    cur_subset_infor = {}
                    cur_subset_infor['subset_attrs'] = list(cur_subset)
                    cur_subset_infor['indices'] = all_valid_subsets_dict[r-1][pre_subsets[i]]['indices'] & all_valid_subsets_dict[r-1][pre_subsets[j]]['indices']
                    cur_subset_infor['sup_num'] = int(np.count_nonzero(cur_subset_infor['indices']))
                    if (cur_subset_infor['sup_num'] < sup_th_num): continue
                    cur_subset_infor['sup_rate'] = round(cur_subset_infor['sup_num'] / residuals.size, 4)
                    # subset_ids_flat = np.where(subset_ids == 1)[0]
                    # subset_multi_ids = np.unravel_index(subset_ids_flat, (n_sample, window_size, n_loc))
                    cur_subset_error = residuals[cur_subset_infor['indices']]
                    subset_pos_errs = cur_subset_error[cur_subset_error >= 0]
                    subset_neg_errs = cur_subset_error[cur_subset_error < 0]
                    cur_subset_infor['residual_abs'] = round(np.mean(np.abs(cur_subset_error)), 4)
                    if err_diff_th != 0 and abs(cur_subset_infor['residual_abs'] - all_valid_subsets_dict[r-1][pre_subsets[i]]['residual_abs']) < err_diff_th or abs(cur_subset_infor['residual_abs'] - all_valid_subsets_dict[r-1][pre_subsets[j]]['residual_abs']) < err_diff_th: continue
                    cur_subset_infor['pos_res_num'] = int(subset_pos_errs.size)
                    cur_subset_infor['neg_res_num'] = int(subset_neg_errs.size)
                    if cur_subset_infor['pos_res_num'] == 0: cur_subset_infor['residual_pos'] = 0
                    else: cur_subset_infor['residual_pos'] = np.mean(subset_pos_errs)
                    if cur_subset_infor['neg_res_num'] == 0: cur_subset_infor['residual_neg'] = 0
                    else: cur_subset_infor['residual_neg'] = np.mean(subset_neg_errs)
                    
                    cur_subset_infor['residual_std'] = np.std(cur_subset_error)
                    cur_subset_infor['residual_entropy'] = entropy(np.abs(cur_subset_error))
                    all_valid_subsets_dict[r][frozenset(cur_subset)] = cur_subset_infor
                    all_subsets.append(cur_subset_infor)
    print('finish step 4')

    # all_subsets = list(subset_indices_dict.values())
    # 为subset构建range_val属性
    for subset in all_subsets:
        subset['range_val'] = {}
        for j in range(len(subset['subset_attrs'])):
            cur_attr_str_split = subset['subset_attrs'][j].split('=')
            cur_attr_type = cur_attr_str_split[0]
            cur_attr_range = eval(cur_attr_str_split[1])
            subset['range_val'][cur_attr_type] = cur_attr_range
    
    sorted_valid_subsets = sorted(all_subsets, key=lambda x: x['residual_abs'], reverse=True)
    for i, subset in enumerate(sorted_valid_subsets):
        # 计算误差分布
        # cur_data_ids = subset['indices_list']
        cur_data_ids = subset['indices']
        cur_residual = residuals[cur_data_ids]
        cur_resi_hist, _ = np.histogram(cur_residual, bins=val_bins, density=False)
        subset['residual_hist'] = cur_resi_hist.tolist()
        subset['residual_hist_normalize'] = np.round(cur_resi_hist / np.max(cur_resi_hist), 6).tolist()
        # 计算属性分布
        subset['subset_id'] = i
        subset['attrs_hist'] = {}
        for attr in all_attrs:
            # cur_attr_data = attr_data[attr][subset['indices_list']]
            cur_attr_data = attr_data[attr][subset['indices']]
            cur_attr_hist, _ = np.histogram(cur_attr_data, bins=attr_objs[attr]['bin_edges'], density=False)
            subset['attrs_hist'][attr] = cur_attr_hist.tolist()
    print('finish step 5')
    # 对valid_subsets根据误差绝对值进行排序
    jsd_th = 0.1
    G = nx.Graph()
    G.add_nodes_from(range(len(all_subsets)))
    redundancy_mtx = np.zeros((len(all_subsets), len(all_subsets)))
    zero_index = val_bins.index(0)
    for i in range(len(all_subsets)):
        for j in range(i+1, len(all_subsets)):
            # 基于jaccard相似性构建重叠度矩阵
            subset_join = sorted_valid_subsets[i]['indices'] & sorted_valid_subsets[j]['indices']
            subset_union = sorted_valid_subsets[i]['indices'] | sorted_valid_subsets[j]['indices']
            redundancy_mtx[i][j] = np.count_nonzero(subset_join) / np.count_nonzero(subset_union)
            redundancy_mtx[j][i] = redundancy_mtx[i][j]
            # 计算子集的误差分布之间的jensen-shannon散度
            pos_resi_hist_i = sorted_valid_subsets[i]['residual_hist_normalize'][zero_index:]
            pos_resi_hist_j = sorted_valid_subsets[j]['residual_hist_normalize'][zero_index:]
            neg_resi_hist_i = sorted_valid_subsets[i]['residual_hist_normalize'][:zero_index]
            neg_resi_hist_j = sorted_valid_subsets[j]['residual_hist_normalize'][:zero_index]
            jsd_pos = jensenshannon(pos_resi_hist_i, pos_resi_hist_j, base=2) ** 2
            jsd_neg = jensenshannon(neg_resi_hist_i, neg_resi_hist_j, base=2) ** 2
            if redundancy_mtx[i][j] > redundancy_th and jsd_pos < jsd_th and jsd_neg < jsd_th:
                G.add_edge(i, j, weight=redundancy_mtx[i][j])
    # print(G)
    # 使用贪心模块化最大化算法进行社区发现
    # independent_set = nx.algorithms.approximation.maximum_independent_set(G)
    # print('independent_set', independent_set)
    # independent_subsets = [all_subsets[i] for i in independent_set]
    # sorted_valid_subsets = sorted(independent_subsets, key=lambda x: x['residual_abs'], reverse=True)
    # print(community.best_partition(G))
    # communities = list(community_louvain.best_partition(G).values())
    # # 查找最大完全联通子图
    # all_cliques = list(nx.find_cliques(G))
    # all_cliques.sort(key=len, reverse=True)
    # print(all_cliques)
    
    # 当图不为空时，才搜索子集社区
    union_subset_list = []
    ids_in_community = []
    if len(G) > 0:
        communities = list(greedy_modularity_communities(G))
        for i, community in enumerate(communities):
            cur_id_list = list(community)
            if len(cur_id_list) == 1: continue
            union_indices = np.zeros(sorted_valid_subsets[0]['indices'].shape, dtype=bool)
            union_subset = {}
            union_subset['contain_subsets'] = []
            union_subset['subset_attrs'] = []
            for subset_id in cur_id_list:
                ids_in_community.append(subset_id)
                union_subset['contain_subsets'].append(sorted_valid_subsets[subset_id])
                cur_indices = sorted_valid_subsets[subset_id]['indices']
                union_indices = union_indices | cur_indices
                union_subset['subset_attrs'] = union_subset['subset_attrs'] + sorted_valid_subsets[subset_id]['subset_attrs']
            union_subset['subset_attrs'] = list(set(union_subset['subset_attrs']))
            union_subset['indices'] = union_indices
            # union_subset['indices_list'] = np.where(union_subset['indices'] == True)[0].tolist()
            union_subset['residual_abs'] = round(np.mean(np.abs(residuals[union_indices])), 4)
            union_subset['sup_num'] = np.where(union_subset['indices'] == True)[0].size
            union_subset['pos_res_num'] = np.count_nonzero(residuals[union_indices] > 0)
            union_subset['neg_res_num'] = np.count_nonzero(residuals[union_indices] < 0)
            union_subset['sup_rate'] = round(union_subset['sup_num'] / residuals.size, 4)
            # 计算属性和误差的分布 - (slice的属性用文字还是hist，感觉可以都用！)
            # 计算误差分布
            cur_residual = residuals[union_indices]
            cur_resi_hist, _ = np.histogram(cur_residual, bins=val_bins, density=False)
            union_subset['residual_hist'] = cur_resi_hist.tolist()
            union_subset['residual_hist_normalize'] = np.round(cur_resi_hist / np.max(cur_resi_hist), 6).tolist()
            # 计算属性分布
            union_subset['subset_id'] = i
            union_subset['attrs_hist'] = {}
            
            for attr in all_attrs:
                cur_attr_data = attr_data[attr][union_indices]
                cur_attr_hist, _ = np.histogram(cur_attr_data, bins=attr_objs[attr]['bin_edges'], density=False)
                union_subset['attrs_hist'][attr] = cur_attr_hist.tolist()
            union_subset_list.append(union_subset)
    # 从原subset中删除在community中的subset，并加入union subset
    
    individual_subsets = [item for i, item in enumerate(sorted_valid_subsets) if i not in ids_in_community]
    # individual_subsets = [item for i, item in enumerate(sorted_valid_subsets)]
    final_subsets = union_subset_list + individual_subsets
    sorted_final_subsets = sorted(final_subsets, key=lambda x: x['residual_abs'], reverse=True)
    
    return sorted_final_subsets