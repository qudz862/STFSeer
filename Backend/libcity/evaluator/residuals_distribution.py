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
from sklearn.tree import DecisionTreeRegressor


class ResidualsDistribution():
    def __init__(self, config, residuals, forecast_scopes):
        self.config = config
        self.distribution_params = self.config.get("distribution_params", {})
        self.residuals = residuals
        self.forecast_scopes = forecast_scopes
        
        self.residual_bins = []
        self.mid_bins = []
        self.pos_extreme_bins = []
        self.neg_extreme_bins = []
        
        self.residual_hists = []
        self.mid_hists = []
        self.pos_extreme_hists = []
        self.neg_extreme_hists = []
        
        self.boxes_infor = []
    
    def divide_extreme_range(self, extreme_residuals):
        attr_data = np.reshape(extreme_residuals, (-1, 1))
        sup_num = math.floor(0.01 * extreme_residuals.size)
        tree = DecisionTreeRegressor(max_leaf_nodes=11, min_samples_leaf=sup_num)
        tree.fit(attr_data, attr_data)
        bin_edges = tree.tree_.threshold[tree.tree_.threshold != -2]
        bin_edges.sort()
        data_min = math.floor(np.min(extreme_residuals))
        data_max = math.ceil(np.max(extreme_residuals))
        all_bin_edges = [data_min]
        all_bin_edges.extend(bin_edges)
        all_bin_edges.append(data_max)
        print('bin_edges with min max:', all_bin_edges)
        bin_edges = [round(bin_edge) for bin_edge in all_bin_edges]
        return bin_edges
    
    def compute_residual_bins(self, residuals=None):
        if residuals is None: residuals = self.residuals
        step_len = int(self.distribution_params['val_step'])
        min_limit = np.min(residuals)
        max_limit = np.max(residuals)
        percentiles = np.percentile(residuals, [25, 50, 75])
        limit_lower_whisker = percentiles[0] - 1.5 * (percentiles[2] - percentiles[0])
        limit_upper_whisker = percentiles[2] + 1.5 * (percentiles[2] - percentiles[0])
        pos_residuals = residuals[residuals > 0]
        neg_residuals = residuals[residuals < 0]        
        limit_lower_out_whisker = np.percentile(neg_residuals, 100-self.distribution_params['extreme_percentile'])
        limit_upper_out_whisker = np.percentile(pos_residuals, self.distribution_params['extreme_percentile'])
        min_limit = -math.ceil(abs(min_limit) / step_len) * step_len
        max_limit = math.ceil(max_limit / step_len) * step_len
        limit_lower_whisker = -math.ceil(abs(limit_lower_whisker) / step_len) * step_len
        limit_upper_whisker = math.ceil(limit_upper_whisker / step_len) * step_len
        limit_lower_out_whisker = -math.ceil(abs(limit_lower_out_whisker) / step_len) * step_len
        limit_upper_out_whisker = math.ceil(limit_upper_out_whisker / step_len) * step_len
        
        if limit_lower_whisker == limit_lower_out_whisker: limit_lower_out_whisker -= step_len
        if limit_upper_whisker == limit_upper_out_whisker: limit_upper_out_whisker += step_len
        
        self.residual_bins = [min_limit, limit_lower_out_whisker]
        self.mid_bins = [limit_lower_out_whisker]
        for bin_edge in range(limit_lower_whisker, limit_upper_whisker+1, step_len):
            self.mid_bins.append(bin_edge)
            self.residual_bins.append(bin_edge)
        self.mid_bins.append(limit_upper_out_whisker)
        self.residual_bins.append(limit_upper_out_whisker)
        self.residual_bins.append(max_limit)
        
        extreme_pos_residuals = pos_residuals[pos_residuals > limit_upper_out_whisker]
        extreme_neg_residuals = neg_residuals[neg_residuals < limit_lower_out_whisker]
        self.pos_extreme_bins = self.divide_extreme_range(extreme_pos_residuals)
        self.neg_extreme_bins = self.divide_extreme_range(extreme_neg_residuals)
        
        # print('residual_bins', self.residual_bins)
        # print('mid_bins', self.mid_bins)
        # print('pos_extreme_bins', self.pos_extreme_bins)
        # print('neg_extreme_bins', self.neg_extreme_bins)
        
        return {
            'residual_bins': self.residual_bins,
            'mid_bins': self.mid_bins,
            'pos_extreme_bins': self.pos_extreme_bins,
            'neg_extreme_bins': self.neg_extreme_bins
            }
    
    def compute_residual_boxes(self, residuals=None, residual_bins=None, forecast_scopes=None):
        # print('residuals', residuals)
        if residuals is None: residuals = self.residuals
        if residual_bins is None: residual_bins = self.residual_bins
        if forecast_scopes is None: forecast_scopes = self.forecast_scopes
        
        self.boxes_infor = []
        for i in range(len(forecast_scopes)):
            scope_residuals = residuals[:,forecast_scopes[i][0]:forecast_scopes[i][1], :]
            percentiles = np.percentile(scope_residuals.flatten(), [25, 50, 75])
            step_len = int(self.distribution_params['val_step'])
            # min_limit = np.min(residuals)
            # max_limit = np.max(residuals)
            # percentiles = np.percentile(residuals, [25, 50, 75])
            limit_lower_whisker = percentiles[0] - 1.5 * (percentiles[2] - percentiles[0])
            limit_upper_whisker = percentiles[2] + 1.5 * (percentiles[2] - percentiles[0])
            # min_limit = -math.ceil(abs(min_limit) / step_len) * step_len
            # max_limit = math.ceil(max_limit / step_len) * step_len
            limit_lower_whisker = -math.ceil(abs(limit_lower_whisker) / step_len) * step_len
            limit_upper_whisker = math.ceil(limit_upper_whisker / step_len) * step_len
            
            mild_pos_outliers = scope_residuals[(scope_residuals > residual_bins[-3]) & (scope_residuals < residual_bins[-2])]
            mild_neg_outliers = scope_residuals[(scope_residuals < residual_bins[2]) & (scope_residuals > residual_bins[1])]
            extreme_pos_outliers = scope_residuals[scope_residuals > residual_bins[-2]]
            extreme_neg_outliers = scope_residuals[scope_residuals < residual_bins[1]]
            
            scope_box_infor = {}
            scope_box_infor['percentiles'] = percentiles.tolist()
            scope_box_infor['lower_whisker'] = int(limit_lower_whisker)
            scope_box_infor['upper_whisker'] = int(limit_upper_whisker)
            scope_box_infor['mild_pos_outliers_num'] = int(mild_pos_outliers.size)
            scope_box_infor['mild_neg_outliers_num'] = int(mild_neg_outliers.size)
            scope_box_infor['extreme_pos_outliers_num'] = int(extreme_pos_outliers.size)
            scope_box_infor['extreme_neg_outliers_num'] = int(extreme_neg_outliers.size)
            print(scope_box_infor)
            # step_err_infor['outliers'] = outliers.tolist()
            self.boxes_infor.append(scope_box_infor)
        return self.boxes_infor
    
    #TODO 根据focus_step_range，提前计算好残差分布 - 包括histogram和box plot的相关数据
    def compute_residual_distribution(self, residuals=None, residual_bins=None, mid_bins=None, pos_extreme_bins=None, neg_extreme_bins=None, forecast_scopes=None):
        # print('residuals', residuals)
        if residuals is None: residuals = self.residuals
        if residual_bins is None: residual_bins = self.residual_bins
        if mid_bins is None: mid_bins = self.mid_bins
        if pos_extreme_bins is None: pos_extreme_bins = self.pos_extreme_bins
        if neg_extreme_bins is None: neg_extreme_bins = self.neg_extreme_bins
        if forecast_scopes is None:
            forecast_scopes = [[0, self.config['output_window']]]
            forecast_scopes += self.forecast_scopes
            print('forecast_scopes', forecast_scopes)
        # for j in range(len(self.config['focus_levels']))
        self.residual_hists = []
        self.mid_hists = []
        self.pos_extreme_hists = []
        self.neg_extreme_hists = []
        for i in range(len(forecast_scopes)):
            # 获取相应的残差数据
            scope_residuals = residuals[:,forecast_scopes[i][0]:forecast_scopes[i][1], :]
            # 计算分布
            residual_hist, _  = np.histogram(scope_residuals.flatten(), bins=residual_bins, density=False)
            self.residual_hists.append(residual_hist.tolist())
            # 计算mid_bins的分布
            mid_residuals = scope_residuals[(scope_residuals < residual_bins[-2])&(scope_residuals > residual_bins[1])]
            mid_hist, _  = np.histogram(mid_residuals, bins=mid_bins, density=False)
            self.mid_hists.append(mid_hist.tolist())
            # 计算pos_extreme_bins的分布
            extreme_pos_outliers = scope_residuals[scope_residuals > residual_bins[-2]]
            pos_extreme_hist, _  = np.histogram(extreme_pos_outliers, bins=pos_extreme_bins, density=False)
            # if self.distribution_params['log_transform']:
            #     pos_extreme_hist = np.log10(pos_extreme_hist + 1e-10)
            self.pos_extreme_hists.append(pos_extreme_hist.tolist())
            # 计算neg_extreme_bins的分布
            extreme_neg_outliers = scope_residuals[scope_residuals < residual_bins[1]]
            neg_extreme_hist, _  = np.histogram(extreme_neg_outliers, bins=neg_extreme_bins, density=False)
            # if self.distribution_params['log_transform']:
            #     neg_extreme_hist = np.log10(neg_extreme_hist + 1e-10)
            self.neg_extreme_hists.append(neg_extreme_hist.tolist())
        
        print('residual_hists', self.residual_hists)
        print('mid_hists', self.mid_hists)
        print('pos_extreme_hists', self.pos_extreme_hists)
        print('neg_extreme_hists', self.neg_extreme_hists)
        
        return {
            'residual_hists': self.residual_hists,
            'mid_hists': self.mid_hists,
            'pos_extreme_hists': self.pos_extreme_hists,
            'neg_extreme_hists': self.neg_extreme_hists
            }
    
    def save_residual_distributions(self, model_name=None):
        if model_name is None: model_name = self.config['model'] + '-' + str(self.config['exp_id'])
        residual_dis_file = f"./error_distributions/{self.config['dataset']}_{model_name}_{self.distribution_params['extreme_percentile']}_{self.forecast_scopes}.npz"
        if not os.path.exists(residual_dis_file):
            np.savez(residual_dis_file, residual_bins=self.residual_bins, mid_bins=self.mid_bins, pos_extreme_bins=self.pos_extreme_bins, neg_extreme_bins=self.neg_extreme_bins, residual_hists=self.residual_hists, mid_hists=self.mid_hists,
            pos_extreme_hists=self.pos_extreme_hists, neg_extreme_hists=self.neg_extreme_hists, boxes_infor=self.boxes_infor)
        
        return 'ok'