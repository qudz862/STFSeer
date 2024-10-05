from libcity.data.famvemd_2d import EMD2DmV
# from libcity.data.famvemd_2d_detrend import EMD2DmV
import importlib
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import copy

from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch, BatchPAD
from scipy.signal import detrend
import time


def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    try:
        return getattr(importlib.import_module('libcity.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        try:
            return getattr(importlib.import_module('libcity.data.dataset.dataset_subclass'),
                           config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')

def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=False,
                        pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader

def generate_dataloader_aug(original_train_data, train_data, eval_data, test_data, feature_name, original_feature_name, batch_size, num_workers, shuffle=False, pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        original_train_data(list of input): 原始训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    original_train_dataset = ListDataset(original_train_data)
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def original_collator(indices):
        batch = Batch(original_feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch
    
    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch

    original_train_dataloader = DataLoader(dataset=original_train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=original_collator,
                                  shuffle=shuffle)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return original_train_dataloader, train_dataloader, eval_dataloader, test_dataloader

def add_gaussian_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def generate_dataloader_mixup(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=False,
                        pad_with_last_sample=False, EMD='no_EMD', imf_num=3, mixup='no_mixup', aug_type='augment'):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。
        mixup: 是否对训练数据进行mixup，以及mixup的类型

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)
    
    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch
    
    def mixup_data(x1, y1, x2, y2, alpha=1.0):
        # print('x1', x1.shape)
        # print('y1', y1.shape)
        # print('x2', x2.shape)
        # print('y2', y2.shape)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        lam = np.array(lam, dtype=np.float32)
        # lam = torch.tensor(lam, dtype=torch.float32)  # 确保 lam 是张量类型
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    train_dataloader = None
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    original_train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    
    if EMD == 'MEMD':
        input_step_num = train_dataset[0][0].shape[0]
        run_num = 5
        param = {
            'nimfs': imf_num,
            'tol': 0.05,
            'type': 6,
            'plot': 'off'
        }       
        print('running MEMD - XD')
        start_time = time.time()
        all_IMFs = []
        all_join_instances = []
        new_seg_instances = []
        for (ins_x, ins_y) in train_dataset:
            # 拼接一个样本的输入与输出
            instance = np.concatenate((ins_x, ins_y), axis=0)
            instance = np.transpose(instance, (2, 1, 0))
            
            all_join_instances.append(instance)
            # 对每个样本进行分解
            # print(np.array(instance).shape)
            result = None
            # ############# EEMD+detrend #############
            # mean_IMFs = np.zeros((instance.shape[0], instance.shape[1], instance.shape[2], imf_num))
            # valid_run_num = 0
            # for i in range(run_num):
            #     noisy_instance = add_gaussian_noise(instance, mean=0, std=0.01)
            #     result = EMD2DmV(noisy_instance, param)
            #     if result != 'fail':
            #         valid_run_num += 1
            #         mean_IMFs += result['IMF']
            #     else:
            #         instance_de = detrend(instance)
            #         trend = instance - instance_de
            #         result = EMD2DmV(instance_de, param)
            #         if result != 'fail':
            #             valid_run_num += 1
            #             for i in range(result['IMF'].shape[3]):
            #                 result['IMF'][:,:,:,i] += trend/imf_num
            #             mean_IMFs += result['IMF']
            # if valid_run_num != 0:
            #     mean_IMFs = mean_IMFs / valid_run_num
            #     all_IMFs.append(mean_IMFs)
            # ############# EEMD+detrend #############
            
            # EMD+直接舍弃
            result = EMD2DmV(instance, param)
            if result != 'fail':
                all_IMFs.append(result['IMF'])
                # 这里直接构建样本？？
                new_instance = np.zeros(all_join_instances[0].shape)
                random_vector = np.random.uniform(0, 2, size=imf_num)
                new_instance = np.tensordot(result['IMF'], random_vector, axes=([3], [0]))
                new_instance = np.transpose(new_instance, (2, 1, 0))
                new_seg_instance = (new_instance[:input_step_num,:,:], new_instance[input_step_num:,:,:])
                new_seg_instances.append(new_seg_instance)
            else:
                new_seg_instances.append((ins_x, ins_y))
        print("get IMFs num: ", len(all_IMFs))
        end_time = time.time()
        run_time = end_time - start_time
        print(f"程序运行时间：{run_time} 秒")
        
        # 构建一个新样本，其每个imf来自不同的样本
        # new_instances = []
        # new_seg_instances = []
        # for i in range(len(all_IMFs)):
        #     new_instance = np.zeros(all_join_instances[0].shape)
        #     random_vector = np.random.uniform(0, 2, size=imf_num)
        #     new_instance = np.tensordot(all_IMFs[i], random_vector, axes=([3], [0]))
            
        #     # for j in range(imf_num):
        #     #     tmp_idx = random.randrange(len(all_IMFs))
        #     #     new_instance = new_instance + all_IMFs[tmp_idx][:,:,:,j]
        #     new_instance = np.transpose(new_instance, (2, 1, 0))
        #     new_seg_instance = (new_instance[:input_step_num,:,:], new_instance[input_step_num:,:,:])
        #     new_instances.append(new_instance)
        #     new_seg_instances.append(new_seg_instance)
        
        # 合成新instance后，再切分成x和y，跟train data合在一起
        # train_dataset = ListDataset(train_data + new_seg_instances)
        train_dataset = ListDataset(new_seg_instances)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    
    if mixup == 'original_mixup':
        print('running original_mixup - XD')
        shuffle_train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=True)
        # 生成 mixup 数据集
        mixed_data = []
        mixed_targets = []

        for batch1, batch2 in zip(train_dataloader, shuffle_train_dataloader):
            mixed_x, mixed_y = mixup_data(np.stack(batch1['X']), np.stack(batch1['y']), np.stack(batch2['X']), np.stack(batch2['y']))
            mixed_data.append(mixed_x)
            mixed_targets.append(mixed_y)

        mixed_data = np.concatenate(mixed_data)
        mixed_targets = np.concatenate(mixed_targets)
        print('mixed_data', mixed_data.shape)
        print('mixed_targets', mixed_targets.shape)

        # 将 mixup 数据集转换为 dataloader
        mixed_dataset = ListDataset(list(zip(mixed_data, mixed_targets)))
        if aug_type == 'augment':
            mixed_dataset = ListDataset(train_data+list(zip(mixed_data, mixed_targets)))
            # x_train, y_train = zip(*list(train_data))
        train_dataloader = DataLoader(dataset=mixed_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    # elif mixup == 'no_mixup':
    #     train_dataloader = train_dataloader
    
    return original_train_dataloader, train_dataloader, eval_dataloader, test_dataloader

def generate_dataloader_subset_aug(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=False,
                        pad_with_last_sample=False, EMD='no_EMD', imf_num=3, mixup='no_mixup', aug_type='augment', scaler=None):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。
        mixup: 是否对训练数据进行mixup，以及mixup的类型

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)
    
    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch
    
    def mixup_data(x1, y1, x2, y2, alpha=1.0):
        # print('x1', x1.shape)
        # print('y1', y1.shape)
        # print('x2', x2.shape)
        # print('y2', y2.shape)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        lam = np.array(lam, dtype=np.float32)
        # lam = torch.tensor(lam, dtype=torch.float32)  # 确保 lam 是张量类型
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    train_dataloader = None
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    original_train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    
    # 构建测试的子集 - 以目标pm2.5范围作为条件
    sub_train_dataset = []
    target_th = (500 - scaler.min) / (scaler.max - scaler.min)
    print(target_th, scaler.max, scaler.min)
    for (ins_x, ins_y) in train_dataset:
        exists_greater = np.any(ins_y[:,:,0] > target_th)
        if exists_greater: sub_train_dataset.append((ins_x, ins_y))
    print('sub_train_dataset', len(sub_train_dataset))
    if EMD == 'MEMD':
        input_step_num = train_dataset[0][0].shape[0]
        run_num = 5
        param = {
            'nimfs': imf_num,
            'tol': 0.05,
            'type': 6,
            'plot': 'off'
        }       
        print('running MEMD - XD')
        start_time = time.time()
        all_IMFs = []
        all_join_instances = []
        new_seg_instances = []
        for (ins_x, ins_y) in sub_train_dataset:
            # 拼接一个样本的输入与输出
            instance = np.concatenate((ins_x, ins_y), axis=0)
            instance = np.transpose(instance, (2, 1, 0))
            
            all_join_instances.append(instance)
            # 对每个样本进行分解
            # print(np.array(instance).shape)
            result = None
            # ############# EEMD+detrend #############
            # mean_IMFs = np.zeros((instance.shape[0], instance.shape[1], instance.shape[2], imf_num))
            # valid_run_num = 0
            # for i in range(run_num):
            #     noisy_instance = add_gaussian_noise(instance, mean=0, std=0.01)
            #     result = EMD2DmV(noisy_instance, param)
            #     if result != 'fail':
            #         valid_run_num += 1
            #         mean_IMFs += result['IMF']
            #     else:
            #         instance_de = detrend(instance)
            #         trend = instance - instance_de
            #         result = EMD2DmV(instance_de, param)
            #         if result != 'fail':
            #             valid_run_num += 1
            #             for i in range(result['IMF'].shape[3]):
            #                 result['IMF'][:,:,:,i] += trend/imf_num
            #             mean_IMFs += result['IMF']
            # if valid_run_num != 0:
            #     mean_IMFs = mean_IMFs / valid_run_num
            #     all_IMFs.append(mean_IMFs)
            # ############# EEMD+detrend #############
            
            # EMD+直接舍弃
            result = EMD2DmV(instance, param)
            if result != 'fail':
                all_IMFs.append(result['IMF'])
                # 这里直接构建样本？？
                new_instance = np.zeros(all_join_instances[0].shape)
                random_vector = np.random.uniform(0, 2, size=imf_num)
                new_instance = np.tensordot(result['IMF'], random_vector, axes=([3], [0]))
                new_instance = np.transpose(new_instance, (2, 1, 0))
                new_seg_instance = (new_instance[:input_step_num,:,:], new_instance[input_step_num:,:,:])
                new_seg_instances.append(new_seg_instance)
            else:
                new_seg_instances.append((ins_x, ins_y))
        print("get IMFs num: ", len(all_IMFs))
        end_time = time.time()
        run_time = end_time - start_time
        print(f"程序运行时间：{run_time} 秒")
        
        # 构建一个新样本，其每个imf来自不同的样本
        # new_instances = []
        # new_seg_instances = []
        # for i in range(len(all_IMFs)):
        #     new_instance = np.zeros(all_join_instances[0].shape)
        #     random_vector = np.random.uniform(0, 2, size=imf_num)
        #     new_instance = np.tensordot(all_IMFs[i], random_vector, axes=([3], [0]))
            
        #     # for j in range(imf_num):
        #     #     tmp_idx = random.randrange(len(all_IMFs))
        #     #     new_instance = new_instance + all_IMFs[tmp_idx][:,:,:,j]
        #     new_instance = np.transpose(new_instance, (2, 1, 0))
        #     new_seg_instance = (new_instance[:input_step_num,:,:], new_instance[input_step_num:,:,:])
        #     new_instances.append(new_instance)
        #     new_seg_instances.append(new_seg_instance)
        
        # 合成新instance后，再切分成x和y，跟train data合在一起
        train_dataset = ListDataset(train_data + new_seg_instances)
        # train_dataset = ListDataset(new_seg_instances)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    
    if mixup == 'original_mixup':
        print('running original_mixup - XD')
        shuffle_train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=True)
        # 生成 mixup 数据集
        mixed_data = []
        mixed_targets = []

        for batch1, batch2 in zip(train_dataloader, shuffle_train_dataloader):
            mixed_x, mixed_y = mixup_data(np.stack(batch1['X']), np.stack(batch1['y']), np.stack(batch2['X']), np.stack(batch2['y']))
            mixed_data.append(mixed_x)
            mixed_targets.append(mixed_y)

        mixed_data = np.concatenate(mixed_data)
        mixed_targets = np.concatenate(mixed_targets)
        print('mixed_data', mixed_data.shape)
        print('mixed_targets', mixed_targets.shape)

        # 将 mixup 数据集转换为 dataloader
        mixed_dataset = ListDataset(list(zip(mixed_data, mixed_targets)))
        if aug_type == 'augment':
            mixed_dataset = ListDataset(train_data+list(zip(mixed_data, mixed_targets)))
            # x_train, y_train = zip(*list(train_data))
        train_dataloader = DataLoader(dataset=mixed_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=False)
    # elif mixup == 'no_mixup':
    #     train_dataloader = train_dataloader
    
    return original_train_dataloader, train_dataloader, eval_dataloader, test_dataloader

def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len(dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = BatchPAD(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    return train_dataloader, eval_dataloader, test_dataloader
