"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_all_path = "F:/PIV_model_generate/PIV_dataset/dataset"

# input: Whether to enter data for all flow conditions; Multiple flow conditions: Yes; Single flow condition: No
all_flow_conditions_used = input("Whether to enter data for all flow conditions; "
                                 "Multiple flow conditions: Yes; "
                                 "Single flow condition: No: ")

if all_flow_conditions_used == "Y":
    try:
        npy_file_names_all = os.listdir(dataset_all_path)
        npy_file_names = npy_file_names_all
    except FileNotFoundError:
        print(f"输入文件夹路径 {dataset_all_path} 不存在。")
        exit(1)
elif all_flow_conditions_used == "N":
    try:
        npy_file_names_all = os.listdir(dataset_all_path)
        npy_file_names = [file for file in npy_file_names_all if file.startswith("PIV-1")]
    except FileNotFoundError:
        print(f"输入文件夹路径 {dataset_all_path} 不存在。")
        exit(1)
else:
    print("输入有误。")
    exit(1)


print(f"流场数据文件的数量为 {len(npy_file_names)}。")

npy_file_names_sorted = sorted(npy_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                               reverse=False)

files_num = len(npy_file_names_sorted)
print(f'流场数据文件的数量为 {files_num}。')

flow_max_list = []
flow_min_list = []

for npy_file in tqdm(npy_file_names_sorted):
    file_path = os.path.join(dataset_all_path, npy_file)
    flow = np.load(file_path, allow_pickle=False)

    flow_max = np.max(flow, axis=(1, 2))
    flow_min = np.min(flow, axis=(1, 2))

    flow_max_list.append(flow_max)
    flow_min_list.append(flow_min)

    # print(f"File: {npy_file}, Max: {flow_max}, Min: {flow_min}")

flow_max_array = np.array(flow_max_list)
flow_min_array = np.array(flow_min_list)

print(flow_min_array.shape)

# Visualize 3 columns of numpy data
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(flow_max_array[:, 0], label="u_max")
ax[0].plot(flow_min_array[:, 0], label="u_min")
ax[0].legend()
ax[0].set_title("u_max and u_min")

ax[1].plot(flow_max_array[:, 1], label="v_max")
ax[1].plot(flow_min_array[:, 1], label="v_min")
ax[1].legend()
ax[1].set_title("v_max and v_min")

ax[2].plot(flow_max_array[:, 2], label="w_max")
ax[2].plot(flow_min_array[:, 2], label="w_min")
ax[2].legend()
ax[2].set_title("w_max and w_min")

plt.show()

# Get the final max-min value from the max-min value in each file.
flow_max_final = np.max(flow_max_array, axis=0)
flow_min_final = np.min(flow_min_array, axis=0)

print(f"Final max: {flow_max_final}, Final min: {flow_min_final}")

# Save the final maximum and minimum values separately as.txt files
if not os.path.exists("..\\data\\minimax"):
    os.makedirs("..\\data\\minimax")

np.savetxt("..\\data\\minimax\\flow_max_final.txt", flow_max_final, fmt="%.12f")
np.savetxt("..\\data\\minimax\\flow_min_final.txt", flow_min_final, fmt="%.12f")



