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
import shutil

import numpy as np
import torch
from torch import nn

import model
from utils import load_state_dict
from natsort import natsorted
from tqdm.auto import tqdm

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

# Load the maximum and minimum values saved locally.
flow_max_final_path = "data/minimax/flow_max_final.txt"
flow_min_final_path = "data/minimax/flow_min_final.txt"

flow_max_final = np.loadtxt(flow_max_final_path)
flow_min_final = np.loadtxt(flow_min_final_path)
flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        # device = torch.device("cuda", 0)
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=2,
                                               out_channels=2,
                                               channels=64)
    sr_model = sr_model.to(device=device)

    return sr_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    try:
        sr_model = load_state_dict(sr_model, args.model_weights_path)
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    # must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    sr_model.eval()

    # Load data
    test_files_name = os.listdir(args.inputs_path)
    test_files_name_sorted = natsorted(test_files_name, reverse=False, )

    file_nums = int(len(test_files_name_sorted))
    print('The number of flow field data files to be reconstructed is {}'.format(file_nums))

    # check the output path, If it exists, delete it and create a new folder
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    for file_name in tqdm(test_files_name_sorted, total=file_nums, desc='Processing Files', ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        file_path = os.path.join(args.inputs_path, file_name)
        try:
            test_lr_flow = np.load(file_path, allow_pickle=False)  # shape [3,H,W]
        except Exception as e:
            print(f"Failed to load `{file_path}`: {e}")
            continue

        # test_lr_flow_normalized = (test_lr_flow[0:2, :, :] - flow_min_final[0:2, :, :]) / (flow_max_final[0:2, :, :] - flow_min_final[0:2, :, :])
        test_lr_flow_normalized = test_lr_flow[0:2, :, :]  # [2,H,W]

        test_lr_flow_normalized = np.ascontiguousarray(test_lr_flow_normalized)

        test_lr_flow_normalized_tensor = torch.from_numpy(test_lr_flow_normalized).float().to(device)
        test_lr_flow_normalized_tensor = test_lr_flow_normalized_tensor.unsqueeze(0)

        # Use the model to generate super-resolved images
        with torch.no_grad():
            test_sr_flow_normalized_tensor = sr_model(test_lr_flow_normalized_tensor)
            test_sr_flow_normalized_tensor_c = test_sr_flow_normalized_tensor.squeeze(0)

        # Save sr_data, shape[C,H,W]
        test_sr_flow_normalized = test_sr_flow_normalized_tensor_c.cpu().numpy()
        test_sr_flow = test_sr_flow_normalized * (flow_max_final[0:2, :, :] - flow_min_final[0:2, :, :]) + flow_min_final[0:2, :, :]
        # test_sr_flow = test_sr_flow_normalized

        save_path = os.path.join(args.output_path, file_name)
        np.save(save_path, test_sr_flow)

    print(f"SR data save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator sr flow datas.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="rdn_small_x4")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/RDN_small_x4-DIV2K-543022e7.pth.tar",
                        help="Model weights file path.")

    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure",
                        help="Low-resolution flow data folder path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure",
                        help="Super-resolution flow data folder path.")

    parser.add_argument("--upscale_factor",
                        type=int,
                        default=4,
                        help="Model upscale factor")

    args = parser.parse_args()

    main(args)
