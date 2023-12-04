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
import argparse
import os

import numpy as np
import torch
from torch import nn

import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64)
    sr_model = sr_model.to(device=device)

    return sr_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_state_dict(sr_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    # must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    sr_model.eval()

    # Load data
    dat_file_names = os.listdir(args.inputs_path)
    dat_file_names_sorted = sorted(dat_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

    extremal_file_names = os.listdir(args.maxmin_path)
    extremal_file_names_sorted = sorted(extremal_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                        reverse=False)

    progression = 1
    File_num = int(len(dat_file_names_sorted))
    print('The number of flow field data files to be reconstructed is {}'.format(File_num))

    for file_id in range(File_num):
        file_path = args.inputs_path + '/' + dat_file_names_sorted[file_id]
        file_path_gt = args.maxmin_path + '/' + extremal_file_names_sorted[file_id]
        fluids_gt_3D = np.load(file_path_gt).astype(np.float32)

        sr_id = int(os.path.splitext(dat_file_names_sorted[file_id].split('_')[1])[0])
        gt_id = int(os.path.splitext(extremal_file_names_sorted[file_id].split('_')[1])[0])
        assert sr_id == gt_id, 'The SR file id num and GT file id num is not equal.'
        # Load data for pre-processing, The data has been normalized to [0,1]
        # lr_tensor = imgproc.preprocess_one_image(args.inputs_path, device)
        lr_tensor, max_xy, min_xy = imgproc.preprocess_one_data(file_path, device)
        max_xy = np.max(fluids_gt_3D, axis=(0, 1))  # One-dimensional arrays
        min_xy = np.min(fluids_gt_3D, axis=(0, 1))

        # Use the model to generate super-resolved images
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor)

        # Save sr_data, shape[H,W,C]
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = sr_image * (max_xy - min_xy) + min_xy

        if progression < 10:
            Num_tag = '0000' + str(progression)
        elif 10 <= progression < 100:
            Num_tag = '000' + str(progression)
        elif 100 <= progression < 1000:
            Num_tag = '00' + str(progression)
        elif 1000 <= progression < 10000:
            Num_tag = '0' + str(progression)
        else:
            Num_tag = str(progression)

        np.save(args.output_path + '/' + 'reconstructedPIV' + '_' + Num_tag + '.npy', sr_image)

        print('Reconstruction calculation processing progress {}/{}'.format(progression, File_num))
        progression += 1

        # sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(args.output_path, sr_image)

    print(f"SR data save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator sr flow datas.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="rdn_small_x4")
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure",
                        help="Low-resolution flow data folder path.")
    parser.add_argument("--maxmin_path",
                        type=str,
                        default="./figure",
                        help="Load the maximum and minimum values corresponding to the data.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure",
                        help="Super-resolution flow data folder path.")
    parser.add_argument("--upscale_factor",
                        type=int,
                        default=4,
                        help="Model upscale factor")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/RDN_small_x4-DIV2K-543022e7.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
