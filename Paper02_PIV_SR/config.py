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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(928)
torch.manual_seed(928)
np.random.seed(928)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
model_arch_name = "rdn_small_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
upscale_factor = 4
# Current configuration parameter method
mode = "train"  # "train"
# Experiment name, easy to save weights and log files
exp_name = "RDN_small_x4-PIV"  # "RDN_small_x4-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_data"

    test_gt_images_dir = f"F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_test"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size_H = int(59-3)  # default: upscale_factor * 32
    gt_image_size_W = int(29-1)
    batch_size = int(16 * 1)  # default:16，越小越好，精度高，但是时间成本很高，论文是16，一开始自己用的64
    num_workers = 4

    # The address to load the pretrained model，预训练模型
    pretrained_model_weights_path = f""

    # Incremental training and migration training，增量训练与迁移训练
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 2000

    # Loss function weight
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = 200 * 2  # default: 200
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 100 * 2

if mode == "test":
    gt_image_size_H = int(59-3)
    gt_image_size_W = int(29-1)
    # Test data address
    lr_dir = f"F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\lr_dir"  # {upscale_factor}
    sr_dir = f"F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\sr_dir"
    gt_dir = f"F:\\model_generator\\No_Heating\\center_blockage_0.7\\test_use\\gt_dir"

    model_weights_path = "./results/RDN_small_x4-PIV/best.pth.tar"
