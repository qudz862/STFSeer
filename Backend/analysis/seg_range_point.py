import sys
sys.path.append('./')
# from libcity.utils import ensure_dir
from collections import OrderedDict
import torch
import pandas as pd

import datetime
import numpy as np
import torch
import math
import os
import json
from collections import Counter
# from main import compute_error_indicators, compute_failure_indicators, compute_focus_indicators
from scipy.stats import entropy  

from sklearn.tree import DecisionTreeRegressor, plot_tree
from libcity.model import loss
import time

# 每个属性划分一个决策树，属性包括: target_val, temporal_state_val, space_diff_val, space_comp_val, pred_step

cur_bin_num = 1

class TreeNode:  
    def __init__(self, attr='', val_range=[0,1], size=0, cover=0, div=0, abs_residual=None, left=None, right=None):  
        self.attr = attr
        self.range = val_range
        self.size = size
        self.cover = cover
        self.div = div
        self.abs_residual = abs_residual
        self.left = left  
        self.right = right

# 遍历二叉树（前序遍历）  
def preorderTraversal(root):  
    if root is None:  
        return []  
    res = []  
    root_dict = {
        'range': root.range,
        'size': root.size,
        'cover': root.cover,
        'div': root.div,
        'abs_residual': root.abs_residual,
        'range_str': root.attr + '=' + str(root.range)
    }
    res.append(root_dict)
    res += preorderTraversal(root.left)  
    res += preorderTraversal(root.right)  
    return res

def get_leaf_nodes(root):
    if root is None:  # 如果根节点为空，返回空列表  
        return []
    root_dict = {
        'range': root.range,
        'size': root.size,
        'cover': root.cover,
        'div': root.div,
        'abs_residual': root.abs_residual,
        'range_str': root.attr + '=' + str(root.range)
    }
    if root.left is None and root.right is None:  # 如果当前节点是叶子节点，返回包含它的列表  
        return [root_dict]  
    else:  # 否则，递归地在左右子树中查找叶子节点  
        return get_leaf_nodes(root.left) + get_leaf_nodes(root.right)

def gen_tree_dict(root, attr, id):
    if root is None:  
        return {}, id
    res = {}
    res['range'] = root.range
    res['size'] = root.size
    res['cover'] = root.cover
    res['div'] = root.div
    res['abs_residual'] = root.abs_residual
    res['range_str'] = attr + '=' + str(id)
    res['children'] = []
    if (root.left is None) and (root.right is None):
        res['range_size'] = root.range[1] - root.range[0]
    if root.left is not None:
        left_tree, id = gen_tree_dict(root.left, attr, id+1)
        res['children'].append(left_tree)
    if root.right is not None:
        right_tree, id = gen_tree_dict(root.right, attr, id+1)
        res['children'].append(right_tree)
    if not res['children']:  # 如果值是一个空列表
        del res['children']
    return res, id

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))

    return H

# 信息熵增益
def gain_entropy(S1_resi, S2_resi, S_resi, event_indicators, global_data_num, err_th):
    S_rate = 1.0 * S_resi.size / global_data_num
    S1_rate = 1.0 * S1_resi.size / global_data_num
    S2_rate = 1.0 * S2_resi.size / global_data_num
    min_resi = event_indicators.min()
    max_resi = event_indicators.max()
    n_bins = int((max_resi - min_resi) / err_th)
    S_hist = np.histogram(S_resi, bins=n_bins, range=(min_resi, max_resi))[0]
    S1_hist = np.histogram(S1_resi, bins=n_bins, range=(min_resi, max_resi))[0]
    S2_hist = np.histogram(S2_resi, bins=n_bins, range=(min_resi, max_resi))[0]
    S_entropy = shan_entropy(S_hist)
    S1_entropy = shan_entropy(S1_hist)
    S2_entropy = shan_entropy(S2_hist)
    gain = S_rate * S_entropy - (S1_rate*S1_entropy + S2_rate*S2_entropy)
    
    return gain

def div_diff(S1_resi, S2_resi, S_resi, err_th):
    mean_abs_S = np.mean(S_resi)
    # S1_div = np.abs(np.mean(np.abs(S1_resi)) - mean_abs_S)
    # S2_div = np.abs(np.mean(np.abs(S2_resi)) - mean_abs_S)
    S1_div = np.abs(np.mean(S1_resi) - mean_abs_S)
    S2_div = np.abs(np.mean(S2_resi) - mean_abs_S)
    if S1_div >= err_th or S2_div >= err_th:
        return True
    else:
        return False

def gain_resi_div(S1_resi, S2_resi, S_resi, global_data_num):
    # S1_rate = 1.0 * S1_resi.size / global_data_num
    # S2_rate = 1.0 * S2_resi.size / global_data_num
    S1_rate = 1.0 * S1_resi.size / S_resi.size
    S2_rate = 1.0 * S2_resi.size / S_resi.size
    gain = S1_rate * np.abs(np.mean(S1_resi) - np.mean(S_resi)) + S2_rate * np.abs(np.mean(S2_resi) - np.mean(S_resi))
    return gain

def find_seg_point(event_attr_data, event_indicators, global_seg_vals, tree_node, range_min, range_max, step_len, sup, err_th):
    step_len = float(step_len)
    range_indices_S = (event_attr_data >= range_min) & (event_attr_data <= range_max)
    # print(event_attr_data.size, event_indicators.size, range_indices_S.size)
    S_resi_abs = event_indicators[range_indices_S]
    
    max_gain = 0
    max_gain_val = None
    max_gain_S1 = None
    max_gain_S2 = None
    # sup是subset针对全局的占比
    max_gain_S1_cover = -1
    max_gain_S2_cover = -1
    max_gain_S1_div = -1
    max_gain_S2_div = -1
    
    global cur_bin_num
    
    attr_range = np.arange(range_min+step_len, range_max-step_len, step_len).tolist()
    for i in range(len(attr_range)):
    # for i in range(int(range_min+step_len), int(range_max-step_len), int(step_len)):
        val = int(attr_range[i] / step_len) * step_len
        range_indices_S1 = (event_attr_data >= range_min) & (event_attr_data <= val)
        range_indices_S2 = (event_attr_data >= val) & (event_attr_data <= range_max)
        # print(S1.size, S2.size, attr_vals.shape)
        S1_resi_abs = event_indicators[range_indices_S1]
        S2_resi_abs = event_indicators[range_indices_S2]
        S1_cover = round((S1_resi_abs.size / event_attr_data.size), 4)
        S2_cover = round((S2_resi_abs.size / event_attr_data.size), 4)
        if S1_cover < sup or S2_cover < sup: continue
        seg_flag_div = div_diff(S1_resi_abs, S2_resi_abs, S_resi_abs, err_th)
        # print('seg_flag_div', seg_flag_div)
        # cur_gain = gain_entropy(S1_pred, S1_truth, S2_pred, S2_truth, S_pred, S_truth, attr_vals.size)
        # cur_gain = gain_resi_div(S1_resi_abs, S2_resi_abs, S_resi_abs, event_attr_data.size)
        cur_gain = gain_entropy(S1_resi_abs, S2_resi_abs, S_resi_abs, event_indicators, event_attr_data.size, err_th)
        # print('cur_gain', cur_gain)
        if seg_flag_div and cur_gain > max_gain:
            max_gain = cur_gain
            max_gain_val = val
            max_gain_S1 = S1_resi_abs
            max_gain_S2 = S2_resi_abs
            max_gain_S1_cover = S1_cover
            max_gain_S2_cover = S2_cover
            
    if max_gain_val is not None:
        startID = global_seg_vals.index(range_min)
        inserted_id = startID + 1
        global_seg_vals.insert(inserted_id, max_gain_val)
        
        max_gain_S1_div = np.mean(max_gain_S1) - np.mean(event_indicators)
        max_gain_S1_abs_residual = np.mean(max_gain_S1)
        tree_node.left = TreeNode(tree_node.attr, [tree_node.range[0], max_gain_val], size=max_gain_S1.size,  cover=max_gain_S1_cover, div=max_gain_S1_div, abs_residual=max_gain_S1_abs_residual)
        global_seg_vals, tree_node.left = find_seg_point(event_attr_data, event_indicators, global_seg_vals, tree_node.left, range_min, max_gain_val, step_len, sup, err_th)
        
        max_gain_S2_div = np.mean(max_gain_S2) - np.mean(event_indicators)
        max_gain_S2_abs_residual = np.mean(max_gain_S2)
        tree_node.right = TreeNode(tree_node.attr, [max_gain_val, tree_node.range[1]], size=max_gain_S2.size, cover=max_gain_S2_cover, div=max_gain_S2_div, abs_residual=max_gain_S2_abs_residual)  
        global_seg_vals, tree_node.right = find_seg_point(event_attr_data, event_indicators, global_seg_vals, tree_node.right, max_gain_val, range_max, step_len, sup, err_th)
        
    return global_seg_vals, tree_node

def seg_range_point(attr, attrs_objs, attr_data, indicators, val_step, sup, err_th):
    seg_points = None
    preorder_tree = None
    tree_dict = None
    indicators_abs = np.abs(indicators).flatten()

    attr_infor = attrs_objs
    range_infor = {}
    preorder_tree = []
    important_nodes = []
    tree_dict = {}
    
    root = TreeNode(attr, attr_infor['range'], size=attr_data.size, cover=1, div=0, abs_residual=float(np.mean(indicators_abs)))
    seg_points, seg_tree = find_seg_point(attr_data, indicators_abs, [attr_infor['range'][0], attr_infor['range'][1]], root, attr_infor['range'][0], attr_infor['range'][1], val_step, sup, err_th)
    
    preorder_tree = preorderTraversal(seg_tree)
    preorder_tree_range = [node['range'] for node in preorder_tree]
    leaf_nodes = get_leaf_nodes(seg_tree)
    # leaf_nodes_range = [node for node in leaf_nodes]
    important_nodes = [node for node in preorder_tree if abs(node['div']) > err_th]
    important_leaf_nodes = [node for node in leaf_nodes if abs(node['div']) > err_th]
    tree_dict, _ = gen_tree_dict(seg_tree, attr, 0)
    range_infor = {
        "seg_points": seg_points,
        "preorder_tree": preorder_tree,
        "preorder_tree_range": preorder_tree_range,
        "leaf_nodes": leaf_nodes,
        "important_nodes": important_nodes,
        "important_leaf_nodes": important_leaf_nodes,
    }
    
    return range_infor