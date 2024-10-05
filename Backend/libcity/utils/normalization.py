import numpy as np
import torch


class Scaler:
    """
    归一化接口
    """

    def transform(self, data):
        """
        数据归一化接口

        Args:
            data(np.ndarray): 归一化前的数据

        Returns:
            np.ndarray: 归一化后的数据
        """
        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):
        """
        数据逆归一化接口

        Args:
            data(np.ndarray): 归一化后的数据

        Returns:
            np.ndarray: 归一化前的数据
        """
        raise NotImplementedError("Inverse_transform not implemented")


class NoneScaler(Scaler):
    """
    不归一化
    """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """
    除以最大值归一化
    x = x / x.max
    """

    def __init__(self, maxx):
        self.max = maxx

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max


class StandardScaler(Scaler):
    """
    Z-score归一化
    x = (x - x.mean) / x.std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMax01Scaler(Scaler):
    """
    MinMax归一化 结果区间[0, 1]
    x = (x - min) / (max - min)
    """

    def __init__(self, minn, maxx):
        self.min = np.nan_to_num(minn, nan=1e-8)
        self.max = np.nan_to_num(maxx, nan=1e-8)

    def transform(self, data):
        trans_data = (data - self.min) / (self.max - self.min)
        np.nan_to_num(trans_data, nan=1e-8)
        return trans_data

    def inverse_transform(self, data):
        return data.cpu() * (torch.from_numpy(self.max) - torch.from_numpy(self.min)) + torch.from_numpy(self.min)


class MinMax11Scaler(Scaler):
    """
    MinMax归一化 结果区间[-1, 1]
    x = (x - min) / (max - min)
    x = x * 2 - 1
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class LogScaler(Scaler):
    """
    Log scaler
    x = log(x+eps)
    """

    def __init__(self, eps=0.999):
        self.eps = eps

    def transform(self, data):
        return np.log(data + self.eps)

    def inverse_transform(self, data):
        return np.exp(data) - self.eps
