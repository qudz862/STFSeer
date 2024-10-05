import torch
import math
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_torch_weighting(preds, labels, weights, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    weighted_loss = loss * weights
    
    return torch.mean(weighted_loss)


def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def masked_mape_torch(preds, labels, null_val=np.nan, eps=0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val) and eps != 0:
        loss = torch.abs((preds - labels) / (labels + eps))
        return torch.mean(loss)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse_torch_weighting(preds, labels, weights, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # print(loss.shape, weights.shape)
    # print(loss.device, weights.device)
    weights = weights.unsqueeze(-1).to('cpu')
    
    weighted_loss = loss * weights
    return torch.mean(weighted_loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val))


def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)


def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)

def acc_score_torch(preds, labels, climatology):
    # global是针对所有地点的，local是针对每个地点的
    acc_global = torch.sum(torch.mul(preds - climatology, labels - climatology)).item() / torch.sqrt(torch.sum(torch.square(preds - climatology))).item() * torch.sqrt(torch.sum(torch.square(labels - climatology))).item()
    acc_local = torch.sum(torch.mul(preds - climatology, labels - climatology), dim=[0,1]) / torch.mul(torch.sqrt(torch.sum(torch.square(preds - climatology), dim=[0,1])), torch.sqrt(torch.sum(torch.square(labels - climatology), dim=[0,1])))
    
    return acc_global, acc_local

def acc_score_np(preds, labels, climatology):
    # global是针对所有地点的，local是针对每个地点的
    acc_global = np.sum((preds - climatology) * (labels - climatology)) / (np.sqrt(np.sum(np.square(preds - climatology))) * np.sqrt(np.sum(np.square(labels - climatology))))
    # acc_local = np.sum((preds - climatology) * (labels - climatology), dim=(0,1)) / (np.sqrt(np.sum(np.square(preds - climatology), dim=[0,1])) * np.sqrt(np.sum(np.square(labels - climatology), dim=(0,1))))
    
    return acc_global

def rmse_np(preds, labels):
    errors = preds - labels
    squared_errors = np.square(errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    return rmse

def space_abs_residual(preds, labels):
    abs_residual = np.abs(preds - labels)
    space_mean = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        space_mean[i] = np.mean(abs_residual[i])
    return space_mean

def compute_hits(preds, labels):
    hits = np.sum((preds == 1) & (labels == 1))
    return hits
def compute_correct_negatives(preds, labels):
    correct_negatives = np.sum((preds == 0) & (labels == 0))
    return correct_negatives
def compute_false_alarms(preds, labels):
    false_alarms = np.sum((preds == 1) & (labels == 0))
    return false_alarms
def compute_misses(preds, labels):
    misses = np.sum((preds == 0) & (labels == 1))
    return misses

def compute_POD(preds, labels):
    hits = compute_hits(preds, labels)
    misses = compute_misses(preds, labels)
    # print(hits, misses)
    if hits == 0:
        pod = 0
    else:
        pod = 1.0 * hits / (hits + misses)
        # print(pod)
    return pod
def compute_FAR(preds, labels):
    hits = compute_hits(preds, labels)
    false_alarms = compute_false_alarms(preds, labels)
    if false_alarms == 0:
        far = 0
    else:
        far = 1.0 * false_alarms / (hits + false_alarms)
    return far
def compute_CSI(preds, labels):
    hits = compute_hits(preds, labels)
    misses = compute_misses(preds, labels)
    false_alarms = compute_false_alarms(preds, labels)
    csi = 1.0 * hits / (hits + misses + false_alarms)
    return csi
def binary_accuracy(preds, labels):
    hits = compute_hits(preds, labels)
    correct_negatives = compute_correct_negatives(preds, labels)
    accuracy = (hits + correct_negatives) / preds.size
    return accuracy
def binary_bias(preds, labels):
    hits = compute_hits(preds, labels)
    false_alarms = compute_false_alarms(preds, labels)
    misses = compute_misses(preds, labels)
    bias = (hits + false_alarms) / (hits + misses)
    return bias

def multi_accuracy_global(preds, labels):
    accurate_num = np.sum(preds == labels)
    if labels.size == 0:
        return 0
    else:
        accuracy = accurate_num / labels.size
        return accuracy

def multi_accuracy_individual(preds, labels, category_num):
    accuracy_list = []
    accurate_num = 0
    for i in range(category_num):
        cur_accurate_num = np.sum((preds == i) & (labels == i))
        cur_whole_num = np.sum((labels == i))
        if (cur_whole_num == 0):
            cur_accuracy = 0
        else:
            cur_accuracy = 1.0 * cur_accurate_num / cur_whole_num
        accuracy_list.append(cur_accuracy)
        accurate_num += cur_accurate_num
    accuracy = accurate_num / preds.size
    return accuracy, accuracy_list
    

