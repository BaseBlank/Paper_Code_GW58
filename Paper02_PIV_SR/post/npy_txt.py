# -*- encoding: utf-8 -*-
'''
@File    :   npy_txt.py   
@Contact :   1574783178@qq.com
@License :   None
 
@Modify Time      @Author       @Version    @Desciption
------------      ----------    --------    -----------
2024/4/18 下午2:50   Liang Biao    1.0         None
'''

import numpy as np
import os
import shutil
from tqdm import tqdm

# Load the maximum and minimum values saved locally.
flow_max_final = np.loadtxt("..\\data\\minimax\\flow_max_final.txt")
flow_min_final = np.loadtxt("..\\data\\minimax\\flow_min_final.txt")
flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]

npy_path = 'F:/PIV_model_generate/PIV_dataset/test_lr_gt'
txt_path = 'F:/PIV_model_generate/PIV_dataset/XYZQ/test_lr_gt_2d_txt'

npy_files = os.listdir(npy_path)

if os.path.exists(txt_path):
    shutil.rmtree(txt_path)
os.makedirs(txt_path)

for file in npy_files:
    file_path = os.path.join(npy_path, file)
    npy_data = np.load(file_path).astype(np.float32)

    C = npy_data.shape[0]
    npy_data = npy_data * (flow_max_final[0:C, :, :] - flow_min_final[0:C, :, :]) + flow_min_final[0:C, :, :]

    velocity_magnitude_matrix = np.zeros((npy_data.shape[1], npy_data.shape[2]))  # (u*82+v**2)*0.5
    for i in range(npy_data.shape[1]):
        for j in range(npy_data.shape[2]):
            velocity_magnitude_matrix[i, j] = np.sqrt(npy_data[0, i, j] ** 2 + npy_data[1, i, j] ** 2)

    np.savetxt(txt_path + '/' + file.replace('.npy', '.txt'), velocity_magnitude_matrix, fmt='%.32f')
