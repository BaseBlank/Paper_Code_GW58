"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================

# Traditional interpolation method realizes conversion of LR to HR data, [H,W,c], 3D -> [rH,rW,c], 3D

import argparse
from tqdm import tqdm
import os
import shutil
from natsort import natsorted
import sys

import numpy as np
import torch

# sys.path.append('F:\\Code\\RDN')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import core


# Load the maximum and minimum values saved locally.
flow_max_final_path = "data/minimax/flow_max_final.txt"
flow_min_final_path = "data/minimax/flow_min_final.txt"

flow_max_final = np.loadtxt(flow_max_final_path)
flow_min_final = np.loadtxt(flow_min_final_path)
flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]


def interpolation_data(args):
    """

    Args:
        args:
    """
    global data_hr_3d
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(args.folder_path_input):
        raise FileNotFoundError(f"The input folder {args.folder_path_input} does not exist.")

    npy_file_names = os.listdir(args.folder_path_input)
    npy_file_names_sorted = natsorted(npy_file_names)

    files_num = int(len(npy_file_names_sorted))
    print('The number of data files to be interpolated is {}'.format(files_num))

    if os.path.exists(args.folder_path_output):
        shutil.rmtree(args.folder_path_output)
    os.makedirs(args.folder_path_output, mode=0o755)

    for file_id in tqdm(npy_file_names_sorted, desc='Interpolating Data'):
        file_path = os.path.join(args.folder_path_input, file_id)
        try:
            data_lr_3d = np.load(file_path).astype(np.float32)  # [H,W,C]
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        if args.interpolation_method == 'gaussian':
            tensor_lr_3d = torch.from_numpy(data_lr_3d).to(device)
            tensor_hr_3d = core.imresize(tensor_lr_3d,
                                         scale=config.upscale_factor,
                                         antialiasing=True,
                                         kernel='gaussian', sigma=1)
            data_hr_3d = tensor_hr_3d.cpu().numpy()

        elif args.interpolation_method == 'cubic':
            tensor_lr_3d = torch.from_numpy(data_lr_3d).to(device)
            tensor_hr_3d = core.imresize(tensor_lr_3d,
                                         scale=config.upscale_factor,
                                         antialiasing=True,
                                         kernel='cubic')
            data_hr_3d = tensor_hr_3d.cpu().numpy()

        elif args.interpolation_method == 'random':
            interpolation_choice = ['gaussian', 'cubic', ]
            interpolation_method = np.random.choice(interpolation_choice)

            if interpolation_method == 'gaussian':
                tensor_lr_3d = torch.from_numpy(data_lr_3d).to(device)
                tensor_hr_3d = core.imresize(tensor_lr_3d,
                                             scale=config.upscale_factor,
                                             antialiasing=True,
                                             kernel='gaussian', sigma=1)
                data_hr_3d = tensor_hr_3d.cpu().numpy()
            elif interpolation_method == 'cubic':
                tensor_lr_3d = torch.from_numpy(data_lr_3d).to(device)
                tensor_hr_3d = core.imresize(tensor_lr_3d,
                                             scale=config.upscale_factor,
                                             antialiasing=True,
                                             kernel='cubic')
                data_hr_3d = tensor_hr_3d.cpu().numpy()

        else:
            raise ValueError('The interpolation method is not supported.')

        data_hr_3d = data_hr_3d * (flow_max_final - flow_min_final) + flow_min_final
        hr_file_save_path = os.path.join(args.folder_path_output, file_id)
        np.save(hr_file_save_path, data_hr_3d)
        # print(data_hr_3d.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the LR data file.")
    parser.add_argument("--folder_path_output", type=str,
                        help="The folder location of interpolation conversion HR data file.")
    parser.add_argument("--interpolation_method", type=str,
                        help="The interpolation method used, including {'gaussian', 'cubic', 'random'}.")

    args = parser.parse_args()

    interpolation_data(args)
