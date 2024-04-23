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
import shutil
from natsort import natsorted
import matplotlib.pyplot as plt

# Load the maximum and minimum values saved locally.
flow_max_final_path = "../data/minimax/flow_max_final.txt"
flow_min_final_path = "../data/minimax/flow_min_final.txt"

flow_max_final = np.loadtxt(flow_max_final_path)
flow_min_final = np.loadtxt(flow_min_final_path)
flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]

# The path of the data file, .npy file
lr_interpolation_path = "F:\\PIV_model_generate\\PIV_dataset\\test_lr_interpolation"
lr_sr_path = "F:\\PIV_model_generate\\PIV_dataset\\test_lr_sr"
lr_gt_path = "F:\\PIV_model_generate\\PIV_dataset\\test_lr_gt"

error_result_path = "F:\\PIV_model_generate\\PIV_dataset\\error_reconstruction"

if os.path.exists(error_result_path):
    shutil.rmtree(error_result_path)
os.makedirs(error_result_path)

files_interpolation = natsorted(os.listdir(lr_interpolation_path))
files_lr_sr = natsorted(os.listdir(lr_sr_path))
files_lr_gt = natsorted(os.listdir(lr_gt_path))

# Ensure all three lists have the same length
assert len(files_interpolation) == len(files_lr_sr) == len(files_lr_gt), \
    "Lists of files have different lengths."

for i in range(len(files_interpolation)):
    interpolation_file = files_interpolation[i]
    lr_sr_file = files_lr_sr[i]
    lr_gt_file = files_lr_gt[i]

    if interpolation_file != lr_sr_file or interpolation_file != lr_gt_file or lr_sr_file != lr_gt_file:
        raise ValueError(f"Mismatch found at index {i}: "
                         f"Interpolation: {interpolation_file}, "
                         f"LR-SR: {lr_sr_file}, "
                         f"LR-GT: {lr_gt_file}")

print("All file names at corresponding indices are identical.")


def uv_to_uvm(uv_array):
    """

    Args:
        uv_array: The array of the velocity field, [2,H,W], 3D

    Returns: The magnitude of the velocity field, [3,H,W], 3D

    """
    # magnitude = (u^2 + v^2)^(1/2)
    u_v_mag = np.ones((3, uv_array.shape[1], uv_array.shape[2]), dtype=np.float32)
    u_v_mag[0, :, :] = uv_array[0, :, :]
    u_v_mag[1, :, :] = uv_array[1, :, :]
    u_v_mag[2, :, :] = np.sqrt(uv_array[0, :, :] ** 2 + uv_array[1, :, :] ** 2)
    # u_v_mag = np.hypot(uv_array[0, :, :], uv_array[1, :, :])

    return u_v_mag


# $$
# \epsilon \equiv \left\| \boldsymbol{x}^{HR}-\mathcal{F} (\boldsymbol{x}) \right\| _2/\left\| \boldsymbol{x}^{HR} \right\| _2
# $$

def calculate_error(component_sr, component_gt):
    """
    Calculate the L2 error for a velocity component.

    Args:
        component_sr: The component of the reconstructed velocity field.
        component_gt: The component of the ground truth velocity field.

    Returns:
        The L2 error for the component.
    """
    return np.linalg.norm(component_sr - component_gt) / np.linalg.norm(component_sr)
    # return np.linalg.norm(component_sr - component_gt) / np.linalg.norm(component_gt)


error_lr_interpolation_record = np.empty((len(files_interpolation), 3), dtype=np.float32)
error_lr_sr_record = np.empty((len(files_lr_sr), 3), dtype=np.float32)

# Create a list to store tuples of (error_mag_lr_sr, file_name)
error_file_list = []

for file_name in files_interpolation:
    matrix_interpolation = np.load(os.path.join(lr_interpolation_path, file_name))[0:2, :, :]  # shape [2,H,W]
    matrix_lr_sr = np.load(os.path.join(lr_sr_path, file_name))  # shape [2,H,W]

    matrix_lr_gt = np.load(os.path.join(lr_gt_path, file_name))[0:2, :, :]  # shape [2,H,W]
    matrix_lr_gt = matrix_lr_gt * (flow_max_final[0:2, :, :] - flow_min_final[0:2, :, :]) + flow_min_final[0:2, :, :]
    # change the shape of the matrix to [2,H,W]
    assert matrix_interpolation.shape == matrix_lr_sr.shape == matrix_lr_gt.shape, \
        "Matrices have different shapes."

    matrix_interpolation_uvm = uv_to_uvm(matrix_interpolation)
    matrix_lr_sr_uvm = uv_to_uvm(matrix_lr_sr)
    matrix_lr_gt_uvm = uv_to_uvm(matrix_lr_gt)

    # check the shape of the matrix
    assert matrix_interpolation_uvm.shape == matrix_lr_sr_uvm.shape == matrix_lr_gt_uvm.shape, \
        "Matrices have different shapes."

    # Extract u and v components from matrices
    u_interpolation, v_interpolation, mag_interpolation = matrix_interpolation_uvm[0], matrix_interpolation_uvm[1], \
        matrix_interpolation_uvm[2]
    u_lr_sr, v_lr_sr, mag_lr_sr = matrix_lr_sr_uvm[0], matrix_lr_sr_uvm[1], matrix_lr_sr_uvm[2]
    u_lr_gt, v_lr_gt, mag_lr_gt = matrix_lr_gt_uvm[0], matrix_lr_gt_uvm[1], matrix_lr_gt_uvm[2]

    # Calculate errors of interpolation for u, v, and velocity magnitude
    error_u_lr_interpolation = calculate_error(u_interpolation, u_lr_gt)
    error_v_lr_interpolation = calculate_error(v_interpolation, v_lr_gt)
    error_mag_lr_interpolation = calculate_error(mag_interpolation, mag_lr_gt)

    # Calculate errors of SR for u, v, and velocity magnitude
    error_u_lr_sr = calculate_error(u_lr_sr, u_lr_gt)
    error_v_lr_sr = calculate_error(v_lr_sr, v_lr_gt)
    error_mag_lr_sr = calculate_error(mag_lr_sr, mag_lr_gt)

    error_lr_interpolation_record[files_interpolation.index(file_name)] = [error_u_lr_interpolation,
                                                                           error_v_lr_interpolation,
                                                                           error_mag_lr_interpolation]
    error_lr_sr_record[files_lr_sr.index(file_name)] = [error_u_lr_sr, error_v_lr_sr, error_mag_lr_sr]

    # Append the error and file name to the list
    error_file_list.append((error_mag_lr_sr, file_name))

    # print(f"L2 Error for u component (LR-Interpolation) for {file_name}: {error_u_lr_interpolation}")
    # print(f"L2 Error for v component (LR-Interpolation) for {file_name}: {error_v_lr_interpolation}")
    # print(f"L2 Error for velocity magnitude (LR-Interpolation) for {file_name}: {error_mag_lr_interpolation}")
    #
    # print(f"L2 Error for u component (LR-SR) for {file_name}: {error_u_lr_sr}")
    # print(f"L2 Error for v component (LR-SR) for {file_name}: {error_v_lr_sr}")
    # print(f"L2 Error for velocity magnitude (LR-SR) for {file_name}: {error_mag_lr_sr}")

# plot the error of the interpolation and SR
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(error_lr_interpolation_record[:, 0], label='u component')
ax[0].plot(error_lr_interpolation_record[:, 1], label='v component')
ax[0].plot(error_lr_interpolation_record[:, 2], label='velocity magnitude')
ax[0].set_title('Interpolation Error')
ax[0].set_xlabel('File Index')
ax[0].set_ylabel('L2 Error')
ax[0].legend()

ax[1].plot(error_lr_sr_record[:, 0], label='u component')
ax[1].plot(error_lr_sr_record[:, 1], label='v component')
ax[1].plot(error_lr_sr_record[:, 2], label='velocity magnitude')
ax[1].set_title('SR Error')
ax[1].set_xlabel('File Index')
ax[1].set_ylabel('L2 Error')
ax[1].legend()

plt.show()

# save the error data as .txt file
np.savetxt(os.path.join(error_result_path, 'L2_error_lr_interpolation.txt'), error_lr_interpolation_record, fmt='%.32f')
np.savetxt(os.path.join(error_result_path, 'L2_error_lr_sr.txt'), error_lr_sr_record, fmt='%.32f')

print('The mean interpolation error.\n u:{}, v:{}, mag:{}'.format(np.mean(error_lr_interpolation_record[:, 0]),
                                                                  np.mean(error_lr_interpolation_record[:, 1]),
                                                                  np.mean(error_lr_interpolation_record[:, 2])))
print('The mean SR error.\n u:{}, v:{}, mag:{}'.format(np.mean(error_lr_sr_record[:, 0]),
                                                       np.mean(error_lr_sr_record[:, 1]),
                                                       np.mean(error_lr_sr_record[:, 2])))

# Sort the list by error (the first element of the tuple), from smallest to largest
error_file_list.sort(key=lambda x: x[0])

# Print the smallest 10 errors and their corresponding file names
print("Smallest 10 L2 errors and corresponding file names:")
for error, file in error_file_list[:10]:
    print(f"{file}: {error}")

# # Assuming 'error_lr_sr_record' is an array where each row corresponds to a file and contains [error_u, error_v, error_mag]
# # and 'files_lr_sr' is a list of file names corresponding to each error record.
#
# # Dictionary to store file names and their corresponding error magnitudes
# error_dict = {file_name: error[2] for file_name, error in zip(files_lr_sr, error_lr_sr_record)}
#
# # Sorting the dictionary by error magnitude in ascending order to get the smallest errors
# sorted_error_dict = sorted(error_dict.items(), key=lambda item: item[1])
#
# # Printing the smallest 10 error magnitudes and their corresponding file names
# print("Smallest 10 error magnitudes and their corresponding file names:")
# for file_name, error_mag in sorted_error_dict[:10]:
#     print(f"{file_name}: {error_mag}")
