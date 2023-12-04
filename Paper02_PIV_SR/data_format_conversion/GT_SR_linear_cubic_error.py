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

Linear_matrix_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/linear_matrix_m"
Cubic_matrix_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/bicubic_matrix_m"
SR_matrix_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR_matrix_m"
GT_matrix_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT_matrix_m"
SR_corr_matrix_path = "F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRDRNNerrorcorr_matrix_m"

file_linear = natsorted(os.listdir(Linear_matrix_path))
file_cubic = natsorted(os.listdir(Cubic_matrix_path))
file_SR = natsorted(os.listdir(SR_matrix_path))
file_GT = natsorted(os.listdir(GT_matrix_path))
file_SR_corr = natsorted(os.listdir(SR_corr_matrix_path))


def array_subtract_mean(array_GT, array_Re):
    """

    Args:
        array_GT: numpy array real results
        array_Re: numpy array reconstruction results

    Returns: Average of difference of two numpy arrays

    """
    if array_GT.shape == array_Re.shape:
        array = abs(array_GT - array_Re) / array_GT
        error_array = (array.sum() / (array.shape[0] * array.shape[1])).astype(np.float32)
    else:
        raise ValueError('The shape of the two arrays is not the same')
    return error_array


def Reconstruction_Results_Error_Cal(real_results_files, reconstruction_results_files,
                                     real_results_folder, reconstruction_results_folder):
    """

    Args:
        reconstruction_results_folder: The folder where the reconstruction result files is located
        real_results_folder: The folder where the real result files is located
        real_results_files: The files where the real result files is located
        reconstruction_results_files: The files where the reconstruction result files is located

    Returns: The average error of the reconstruction results of all test cases under the specific algorithm

    """

    file_num = len(real_results_files)

    Re_error_record = np.empty((file_num, 1), dtype=np.float32)

    for file_id in range(file_num):
        if os.path.splitext(real_results_files[file_id].split('_')[1])[0] == \
                os.path.splitext(reconstruction_results_files[file_id].split('_')[1])[0]:
            data_GT = np.loadtxt(real_results_folder + '/' + real_results_files[file_id], dtype=np.float32,
                                 delimiter=' ')
            data_SR = np.loadtxt(reconstruction_results_folder + '/' + reconstruction_results_files[file_id],
                                 dtype=np.float32, delimiter=' ')

            Re_error = array_subtract_mean(array_GT=data_GT, array_Re=data_SR)
            Re_error_record[file_id, :] = Re_error

        else:
            print("Error: The GT file name is not match SR file name.")

    # Arrange the elements in the array Re_error_record from small to large
    Re_error_record_sorted = np.sort(Re_error_record, axis=0, kind='mergesort')
    # 去除LR下采样后由于取得的极值偏离原始真实数据的极值，带来的明显偏大的重建误差
    Re_error_record_final = Re_error_record_sorted[0:int(0.9 * file_num), :]
    return Re_error_record.mean().astype(np.float32), Re_error_record_final.mean().astype(np.float32), Re_error_record


error_result_path = "F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\2DArray\\error_matrix_m"

error_linear_mean, error_linear_mean_opt, error_linear = Reconstruction_Results_Error_Cal(real_results_files=file_GT,
                                                                                          reconstruction_results_files=file_linear,
                                                                                          real_results_folder=GT_matrix_path,
                                                                                          reconstruction_results_folder=Linear_matrix_path)
print('The error of the reconstruction result of the one level linear algorithm is \n {} - {}'.format(error_linear_mean,
                                                                                                      error_linear_mean_opt))
np.savetxt(fname=error_result_path + '/' + 'error_linear' + '.txt', X=error_linear, fmt='%.32f')

error_cubic_mean, error_cubic_mean_opt, error_cubic = Reconstruction_Results_Error_Cal(real_results_files=file_GT,
                                                                                       reconstruction_results_files=file_cubic,
                                                                                       real_results_folder=GT_matrix_path,
                                                                                       reconstruction_results_folder=Cubic_matrix_path)
print('The error of the reconstruction result of the third order cubic algorithm is \n {} - {}'.format(error_cubic_mean,
                                                                                                       error_cubic_mean_opt))
np.savetxt(fname=error_result_path + '/' + 'error_cubic' + '.txt', X=error_cubic, fmt='%.32f')

error_SR_mean, error_SR_mean_opt, error_SR = Reconstruction_Results_Error_Cal(real_results_files=file_GT,
                                                                              reconstruction_results_files=file_SR,
                                                                              real_results_folder=GT_matrix_path,
                                                                              reconstruction_results_folder=SR_matrix_path)
print('The error of the reconstruction result of the neural networks algorithm is \n {} - {}'.format(error_SR_mean,
                                                                                                     error_SR_mean_opt))
np.savetxt(fname=error_result_path + '/' + 'error_SR' + '.txt', X=error_SR, fmt='%.32f')

error_SR_corr_mean, error_SR_corr_mean_opt, error_SR_corr = Reconstruction_Results_Error_Cal(real_results_files=file_GT,
                                                                                             reconstruction_results_files=file_SR_corr,
                                                                                             real_results_folder=GT_matrix_path,
                                                                                             reconstruction_results_folder=SR_corr_matrix_path)
print('The error of the reconstruction result of the corred neural networks algorithm is \n {} - {}'.format(
    error_SR_corr_mean, error_SR_corr_mean_opt))
np.savetxt(fname=error_result_path + '/' + 'error_SR_corr' + '.txt', X=error_SR_corr, fmt='%.32f')
