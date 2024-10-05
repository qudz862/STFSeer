import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import rank_filter
from scipy.stats import mode
from scipy.ndimage import uniform_filter
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet

def EMD2DmV(data, param):
    # print('data', data)
    # 获取数据的维度
    num_vars, Nx, Ny = data.shape

    # 初始化参数
    if not 'nimfs' in param:
        param['nimfs'] = 10
    if not 'tol' in param:
        param['tol'] = 0.05
    if not 'type' in param:
        param['type'] = 6
    if not 'plot' in param:
        param['plot'] = 'off'
    
    if not all(x in [1, 2, 3, 4, 5, 6, 7] for x in [param['type']]):
        raise ValueError('Please enter a valid window size type')

    if param['tol'] <= 0.005:
        print('Low sifting tolerance may cause oversifting')
        answer = input('Would you like to continue? [Yes/No]: ')
        if answer.lower() == 'no':
            return
    
    IMF = np.zeros((num_vars, Nx, Ny, param['nimfs']))
    Residue = data.copy()
    # print('Residue', Residue)
    Windows = np.zeros((7, param['nimfs']))
    sift_cnt = np.zeros(param['nimfs'])

    imf = 1
    stopflag = True

    while imf <= param['nimfs'] and stopflag:
        H = Residue.copy()
        sift_stop = False

        Combined = np.mean(H, axis=0) / np.sqrt(num_vars)
        maxima, minima = identify_max_min(Combined)
        # print('maxima, minima', maxima.shape, minima.shape)
        
        if np.count_nonzero(maxima) < 3 or np.count_nonzero(minima) < 3:
            # print('Fewer than three extrema found in maxima map. Stopping now...')
            return 'fail'
            break

        Windows[:, imf - 1] = filter_size(maxima, minima, param['type'])
        # print(Windows[:, imf - 1])
        w_sz = Windows[param['type'] - 1, imf - 1]
        
        if not w_sz:
            # print('w_sz', w_sz)
            # print('EMD2D3V has stopped because the Delaunay Triangulation could not be created (collinear points)')
            stopflag = False
            return 'fail'
            break
        while not sift_stop:
            # print('H', H)
            sift_cnt[imf - 1] += 1
            # print('H1', H)
            Env = osf(H, w_sz)
            # print('Env', Env)
            Env = pad_smooth(Env, w_sz)
            Env_med = np.stack([Env[var]['maxs'] + Env[var]['mins'] for var in range(num_vars)]) / 2
            
            H1 = H - Env_med
            
            mse = np.mean((H1 - H) ** 2, axis=(1, 2))
            # print(mse)
            
            if np.all(mse < param['tol']) and sift_cnt[imf - 1] != 1:
                sift_stop = True
            
            H = H1
        
        IMF[:, :, :, imf - 1] = H
        Residue -= IMF[:, :, :, imf - 1]
        
        imf += 1
    
    if any(sift_cnt >= 5 * np.ones(sift_cnt.shape)):
        print('Decomposition may be oversifted. Checking if window size increases monotonically...')
        if any(np.diff(Windows[param['type'] - 1, :]) <= 0):
            print('Filter window size does not increase monotonically')
    
    Results = {
        'IMF': IMF,
        'windowtype': param['type'],
        'Residue': Residue,
        'Windows': Windows,
        'Sifts': sift_cnt,
        'IO': [None] * num_vars,
        'Error': [None] * num_vars
    }
    
    # for i in range(num_vars):
    #     Results['IO'][i], Results['Error'][i] = orth_index(data[i], IMF[i], Residue[i])
    
    if param['plot'] == 'on':
        plot_results(data, Results, param)
    
    return Results

epsilon = 1e-9
def identify_max_min(signal):
    mask = np.ones((3, 3), dtype=np.int32)
    mask[1][1] = 0
    # 使用 rank_filter 执行排序滤波
    B = rank_filter(signal, -1, footprint=mask)
    C = rank_filter(signal, 0, footprint=mask)
    # print('B C', B.shape, C.shape)
    # 生成最大值和最小值的布尔数组
    maxima = signal >= B
    minima = signal <= C
    if np.count_nonzero(maxima) <= 2:
        mask = np.ones((3, 3), dtype=np.int32)
        mask[1][1] = 0
        mask[0][1] = 0
        mask[1][0] = 0
        mask[2][1] = 0
        mask[1][2] = 0
        # 使用 rank_filter 执行排序滤波
        B = rank_filter(signal, -1, footprint=mask)
        C = rank_filter(signal, 0, footprint=mask)
        # print('B C', B.shape, C.shape)
        # 生成最大值和最小值的布尔数组
        maxima = signal >= B
        minima = signal <= C
    if np.count_nonzero(minima) <= 2:
        mask = np.ones((3, 3), dtype=np.int32)
        mask[1][1] = 0
        mask[0][1] = 0
        mask[1][0] = 0
        mask[2][1] = 0
        mask[1][2] = 0
        # 使用 rank_filter 执行排序滤波
        B = rank_filter(signal, -1, footprint=mask)
        C = rank_filter(signal, 0, footprint=mask)
        # print('B C', B.shape, C.shape)
        # 生成最大值和最小值的布尔数组
        maxima = signal >= B
        minima = signal <= C
        # print('minima', np.count_nonzero(minima))
        # print('maxima', np.count_nonzero(maxima))
    
    return maxima, minima

def filter_size(maxima_map, minima_map, type):
    maxima_pos = np.array(np.nonzero(maxima_map)).T
    minima_pos = np.array(np.nonzero(minima_map)).T

    # print(f"Maxima positions shape: {maxima_pos.shape}")
    # print(f"Minima positions shape: {minima_pos.shape}")
    
    try:
        TRI_max = Delaunay(maxima_pos)
        # print('maxima_pos', maxima_pos)
    except:
        # print('Maxima points are collinear. Exiting without further iterations')
        return np.zeros(7)
    
    e1 = np.sqrt((maxima_pos[TRI_max.simplices[:, 1], 0] - maxima_pos[TRI_max.simplices[:, 0], 0]) ** 2 + 
                 (maxima_pos[TRI_max.simplices[:, 1], 1] - maxima_pos[TRI_max.simplices[:, 0], 1]) ** 2)
    e2 = np.sqrt((maxima_pos[TRI_max.simplices[:, 2], 0] - maxima_pos[TRI_max.simplices[:, 0], 0]) ** 2 + 
                 (maxima_pos[TRI_max.simplices[:, 2], 1] - maxima_pos[TRI_max.simplices[:, 0], 1]) ** 2)
    e3 = np.sqrt((maxima_pos[TRI_max.simplices[:, 2], 0] - maxima_pos[TRI_max.simplices[:, 1], 0]) ** 2 + 
                 (maxima_pos[TRI_max.simplices[:, 2], 1] - maxima_pos[TRI_max.simplices[:, 1], 1]) ** 2)

    em1 = np.min(np.stack((e2, e1), axis=1), axis=1)
    em2 = np.min(np.stack((e3, e1), axis=1), axis=1)
    em3 = np.min(np.stack((e3, e2), axis=1), axis=1)
    
    max_nearest = np.zeros(maxima_pos.shape[0])
    e = np.vstack((em1, em2, em3)).T

    for i in range(len(em1)):
        for j in range(3):
            if max_nearest[TRI_max.simplices[i, j]] > e[i, j] or max_nearest[TRI_max.simplices[i, j]] == 0:
                max_nearest[TRI_max.simplices[i, j]] = e[i, j]

    try:
        TRI_min = Delaunay(minima_pos)
        # print('minima_pos', minima_pos)
    except:
        # print('Minima points are collinear. Exiting without further iterations')
        return np.zeros(7)
    
    e1 = np.sqrt((minima_pos[TRI_min.simplices[:, 1], 0] - minima_pos[TRI_min.simplices[:, 0], 0]) ** 2 + 
                 (minima_pos[TRI_min.simplices[:, 1], 1] - minima_pos[TRI_min.simplices[:, 0], 1]) ** 2)
    e2 = np.sqrt((minima_pos[TRI_min.simplices[:, 2], 0] - minima_pos[TRI_min.simplices[:, 0], 0]) ** 2 + 
                 (minima_pos[TRI_min.simplices[:, 2], 1] - minima_pos[TRI_min.simplices[:, 0], 1]) ** 2)
    e3 = np.sqrt((minima_pos[TRI_min.simplices[:, 2], 0] - minima_pos[TRI_min.simplices[:, 1], 0]) ** 2 + 
                 (minima_pos[TRI_min.simplices[:, 2], 1] - minima_pos[TRI_min.simplices[:, 1], 1]) ** 2)

    em1 = np.minimum(e2, e1)
    em2 = np.minimum(e3, e1)
    em3 = np.minimum(e3, e2)
    
    min_nearest = np.zeros(minima_pos.shape[0])
    e = np.vstack((em1, em2, em3)).T

    for i in range(len(em1)):
        for j in range(3):
            if min_nearest[TRI_min.simplices[i, j]] > e[i, j] or min_nearest[TRI_min.simplices[i, j]] == 0:
                min_nearest[TRI_min.simplices[i, j]] = e[i, j]

    d1 = np.min([np.min(max_nearest) , np.min(min_nearest)])
    d2 = np.max([np.min(max_nearest) , np.min(min_nearest)])
    d3 = np.min([np.max(max_nearest) , np.max(min_nearest)])
    d4 = np.max([np.max(max_nearest) , np.max(min_nearest)])
    d5 = (d1+d2+d3+d4)/4
    combined_data = np.concatenate((min_nearest, max_nearest))
    d6 = np.median(combined_data)
    d7 = mode(combined_data, keepdims=True)[0][0]  # 返回众数的值，注意众数可能不唯一
    
    Windows = np.array([d1, d2, d3, d4, d5, d6, d7])

    # 确保 w_size 是一个奇数整数
    Windows = 2 * (np.floor(Windows / 2)) + 1
    if(Windows[type-1]<3):
    #   print('WARNING: Calculated Window size less than 3')
    #   print('Overriding calculated value and setting window size = 3')
      Windows[type-1] = 3
    
    return Windows
cnt = 1
# 顺序统计过滤以确定最大和最小包络
def osf(H, w_sz):
    mask = np.ones((int(w_sz), int(w_sz)), dtype=bool)
    
    Env = {}
    for var in range(H.shape[0]):
        # print('H[var]', H)
        Env[var] = {'maxs': rank_filter(H[var], -1, footprint=mask),
                    'mins': rank_filter(H[var], 0, footprint=mask)}
        # Env[var]['mins'] = -Env[var]['mins']
    # global cnt
    # if cnt <= 2: print(Env)
    # cnt += 1
    return Env

def moving_mean(array, size):
    return uniform_filter(array, size=size, mode='reflect', cval=0.0)

def pad_smooth(Env, w_sz):
    h = int(w_sz // 2)
    padded_Env = {}
    for var in Env.keys():
        # print('cur_pad_max1', Env[var]['maxs'].shape)
        # cur_pad_max = np.pad(Env[var]['maxs'], ((h, h), (h, h)), mode='edge')
        # cur_pad_min = np.pad(Env[var]['mins'], ((h, h), (h, h)), mode='edge')
        # print('cur_pad_max', cur_pad_max.shape)
        temp_max = moving_mean(Env[var]['maxs'], size=(1, w_sz))
        temp_min = moving_mean(Env[var]['mins'], size=(1, w_sz))
        padded_Env[var] = {'maxs': moving_mean(temp_max, size=(w_sz, 1)),
                           'mins': moving_mean(temp_min, size=(w_sz, 1))}
    return padded_Env

def orth_index(data, IMF, Residue):
    # Orthogonality index and error calculation
    orth_idx = np.zeros(IMF.shape[2])
    error = np.zeros(IMF.shape[2])
    
    for i in range(IMF.shape[2]):
        imf_i = IMF[:, :, i]
        for j in range(i + 1, IMF.shape[2]):
            imf_j = IMF[:, :, j]
            orth_idx[i] += np.sum(imf_i * imf_j) / (np.linalg.norm(imf_i) * np.linalg.norm(imf_j))
        
        error[i] = np.sum((data - (np.sum(IMF[:, :, :i+1], axis=2) + Residue))**2) / np.sum(data**2)
    
    return orth_idx, error

def plot_results(data, Results, param):
    # Plotting results
    num_vars = data.shape[0]
    nimfs = param['nimfs']
    
    fig, axs = plt.subplots(num_vars, nimfs + 1, figsize=(15, num_vars * 3))
    
    for var in range(num_vars):
        axs[var, 0].imshow(data[var], cmap='gray')
        axs[var, 0].set_title(f'Original Data {var+1}')
        
        for imf in range(nimfs):
            axs[var, imf + 1].imshow(Results['IMF'][var, :, :, imf], cmap='gray')
            axs[var, imf + 1].set_title(f'IMF {imf+1}')
    
    plt.tight_layout()
    plt.show()
    
# # 示例用法
# time_points = 50
# space_points = 50
# variables = 3

# t = np.linspace(0, 1, time_points)
# x = np.sin(2 * np.pi * 3 * t) + 0.5 * np.random.randn(time_points)
# y = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(time_points)
# z = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(time_points)

# # 生成二维时空数据
# data_x = np.tile(x, (space_points, 1)).T
# data_y = np.tile(y, (space_points, 1)).T
# data_z = np.tile(z, (space_points, 1)).T

# # 组合成多变量二维时空数据
# multivariate_signal = np.array([data_x, data_y, data_z])
# # print(multivariate_signal)

# data = np.random.rand(3, 100, 100)  # 假设我们有3个变量，每个变量是128x128的图像
# # print(data)
# # print(data.shape, multivariate_signal.shape)
# param = {
#     'nimfs': 7,
#     'tol': 0.05,
#     'type': 6,
#     'plot': 'on'
# }
# results = EMD2DmV(multivariate_signal, param)
# # 生成新的样本？？
# print(results['IMF'].shape)