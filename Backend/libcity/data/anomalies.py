import os
import torch
import math
import numpy as np
import pandas as pd
import json
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class LitAutoencoder(pl.LightningModule):
    def __init__(self, input_dim):
        super(LitAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class Anomalies():
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data
        self.anomaly_types = self.config.get('anomaly_types', [])
    
    def call_method_by_name(self, method_name, *args, **kwargs):
        cur_metadata = None
        # 使用 getattr 获取方法，并调用该方法
        method = getattr(self, method_name, None)
        if callable(method):
            cur_metadata = method(*args, **kwargs)
        else:
            print(f"Method '{method_name}' not found")
        return cur_metadata
    
    def z_score_anomaly(self, z_th=3):
        mean = np.mean(self.raw_data[..., 0])
        std = np.std(self.raw_data[..., 0])
        z_scores = (self.raw_data[..., 0] - mean) / std
        anomalies = np.abs(z_scores) > z_th
        return anomalies
    
    def neighbor_anomaly(self, dis_th=25, rate_th=0.5):
        n_step, n_loc, n_var = self.raw_data.shape
        # load distance relations
        rel_file = './raw_data/' + self.config['dataset'] + '/' + self.config['dataset'] + '.rel'
        rel_df = pd.read_csv(rel_file)
        # compute distance matrix
        dis_mtx = np.zeros((n_loc, n_loc))
        for i in range(n_loc):
            for j in range(n_loc):
                if i == j: continue
                df_index = i * (n_loc-1) + j
                dis_mtx[i][j] = rel_df.loc[df_index, 'distance']
        
        all_neighbor_means = np.zeros((n_step, n_loc))
        for i in range(n_step):
            nbrs = NearestNeighbors(radius=dis_th, metric='precomputed').fit(dis_mtx)
            distances, indices = nbrs.radius_neighbors(dis_mtx, dis_th)
            # 记录每个点的邻域内数值和
            neighbor_mean = np.array([np.mean(self.raw_data[i,neighbors,0]) for neighbors in indices])
            all_neighbor_means[i] = neighbor_mean
        # 判断数值和超过阈值的点为异常点
        anomalies = self.raw_data[...,0] - (1 + rate_th) * all_neighbor_means > 0
        
        return anomalies
    
    # extract temporal anomaly based on moving window
    def temporal_anomaly(self, window_size=12, std_th=2):
        # get time series for every locations
        time_series = np.transpose(self.raw_data[...,0], (1,0))
        n_loc, n_step = time_series.shape
        all_temporal_anomalies = np.zeros((n_loc, n_step), dtype=bool)
        for i in range(n_loc):
            # compute mean and std in moving window
            moving_avg = np.convolve(time_series[i], np.ones(window_size)/window_size, mode='valid')
            moving_std = np.array([np.std(time_series[i][j:j+window_size]) for j in range(n_step - window_size + 1)])
            seq_anomalies = np.abs(time_series[i][window_size-1:] - moving_avg) > std_th * moving_std
            all_temporal_anomalies[i, :n_step-window_size+1] = seq_anomalies
        all_temporal_anomalies = np.transpose(all_temporal_anomalies, (1,0))
        
        return all_temporal_anomalies

    def autoencoder_anomaly(self, threshold=0.01):
        # construct dataloader
        n_step, n_loc, n_var = self.raw_data.shape
        flattened_data = np.reshape(self.raw_data, (n_step*n_loc, n_var))
        data_tensor = torch.tensor(flattened_data, dtype=torch.float32)
        batch_size = 64
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LitAutoencoder(input_dim=n_var)

        # use the Trainer of PyTorch Lightning to train the data
        trainer = pl.Trainer(max_epochs=100)
        trainer.fit(model, dataloader)
        model.eval()
        
        with torch.no_grad():
            reconstructed_data = model(data_tensor).numpy()

        # use MSE to compute the reconstruction error
        reconstruction_error = np.mean((flattened_data - reconstructed_data) ** 2, axis=1)

        # Set the 95th percentile as the anomaly detection threshold
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        return anomalies
    
    def cluster_anomaly(self, min_samples_rate=0.0001, max_eps=np.inf, xi=0.05):
        min_samples = int(min_samples_rate * self.raw_data.size)
        n_step, n_loc, n_var = self.raw_data.shape
        # Flatten spatial dimensions for clustering
        flattened_data = np.reshape(self.raw_data, (n_step*n_loc, n_var))
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi).fit(flattened_data)
        
        labels = optics.labels_
        # Outliers are labeled as -1 in OPTICS
        anomalies = labels == -1
        
        # Reshape back to original spatial dimensions
        anomalies = anomalies.reshape(n_step, n_loc)
        return anomalies
        
    def extract_anomalies(self):
        # 先判断异常数据文件是否已存在
        anomalies_dir = f"./anomalies_data/{self.config['dataset']}/"
        if not os.path.exists(anomalies_dir):
            os.makedirs(anomalies_dir)
        params_str = '_'.join(self.anomaly_types)
        anomalies_file = f"{anomalies_dir}/anomalies_{params_str}.npz"
        if os.path.exists(anomalies_file):
            print("anomalies file already existed.")
            return 'ok'
        all_anomaly_flags = {}
        all_anomaly_indices = {}
        for attr in self.anomaly_types:
            print('cur anomaly type: ', attr)
            anomaly_flags = self.call_method_by_name(attr)
            all_anomaly_flags[attr] = anomaly_flags
            indices = np.where(anomaly_flags == True)
            all_anomaly_indices[attr] = np.array(list(zip(indices[0], indices[1])))
        np.savez(anomalies_file, **all_anomaly_indices)
        
        return 'ok'