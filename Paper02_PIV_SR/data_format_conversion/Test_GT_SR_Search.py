"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""
# ==============================================================================
# Convert the three-dimensional tset data calculated by neural network reasoning into two-dimensional data

import numpy as np
import os
from natsort import natsorted

GT_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT"
SR_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR"

file_npy_GT = natsorted(os.listdir(GT_path))
file_npy_SR = natsorted(os.listdir(SR_path))

file_num = len(file_npy_GT)
print(file_npy_GT)

error_file_sorted = []

for file_id in range(file_num):
    if os.path.splitext(file_npy_GT[file_id].split('_')[1])[0] == os.path.splitext(file_npy_SR[file_id].split('_')[1])[0]:
        data_GT = np.loadtxt(GT_path + '/' + file_npy_GT[file_id], dtype=np.float32, delimiter=' ')
        data_SR = np.loadtxt(SR_path + '/' + file_npy_SR[file_id], dtype=np.float32, delimiter=' ')

        Re_error = np.empty((data_GT.shape[0],), dtype=np.float32)
        for i in range(data_GT.shape[0]):
            Re_error[i] = abs((data_GT[i, 5] - data_SR[i, 5]) / data_GT[i, 5])
        Re_error_sum = np.sum(Re_error)
        error_file_list = os.path.splitext(file_npy_GT[file_id])[0] + '-' + str(Re_error_sum)
        error_file_sorted.append(error_file_list)

    else:
        print("Error: The GT file name is not match SR file name.")

Error_File_Sorted = sorted(error_file_sorted, key=lambda name: float(name.split('-')[1]), reverse=False)

# print(error_file_sorted)
print('误差最小的前10个重建文件为 \n {}'.format(Error_File_Sorted[0:10]))

extra_calculation = input("Whether to perform additional calculations to save the various algorithms？(y/n): ")

if extra_calculation == 'y':
    # Save the two-dimensional distribution matrix of several reconstruction files with the smallest error
    save_path = 'F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\error_matrix_m'
    GT_matrix_path = \
        'F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\GT_matrix_m\\2DMatrixGTm_02577.txt'
    linear_matrix_path = \
        'F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\linear_matrix_m\\linearm_02577.txt'
    bicubic_matrix_path = \
        'F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\bicubic_matrix_m\\bicubicm_02577.txt'
    SR_matrix_path = \
        'F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\SR_matrix_m\\2DMatrixSRm_02577.txt'

    GT_matrix = np.loadtxt(GT_matrix_path, dtype=np.float32, delimiter=' ')
    linear_matrix = np.loadtxt(linear_matrix_path, dtype=np.float32, delimiter=' ')
    bicubic_matrix = np.loadtxt(bicubic_matrix_path, dtype=np.float32, delimiter=' ')
    SR_matrix = np.loadtxt(SR_matrix_path, dtype=np.float32, delimiter=' ')

    SR_error_matrix = abs(GT_matrix - SR_matrix)
    linear_error_matrix = abs(GT_matrix - linear_matrix)
    bicubic_error_matrix = abs(GT_matrix - bicubic_matrix)

    np.savetxt(save_path + '/SR_error_matrix.txt', SR_error_matrix, fmt='%.32f', delimiter=' ')
    np.savetxt(save_path + '/linear_error_matrix.txt', linear_error_matrix, fmt='%.32f', delimiter=' ')
    np.savetxt(save_path + '/bicubic_error_matrix.txt', bicubic_error_matrix, fmt='%.32f', delimiter=' ')

    print("The error matrix of the three reconstruction algorithms has been saved.")

else:
    print("The error matrix of the three reconstruction algorithms has not been saved.")


