# -*- encoding: utf-8 -*-
"""
@File    :   GT_LR_generate.py
@Contact :   1574783178@qq.com
@License :   None

@Modify Time      @Author       @Version    @Desciption
------------      ----------    --------    -----------
21/3/2024 下午9:18   Liang Biao    1.0         None
"""

# ==============================================================================
import argparse
import os
import shutil
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import multiprocessing

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import imgproc
import torch
import core

# random.seed(928)


def main(args) -> None:
    try:
        if os.path.exists(args.lr_dir_normalize):
            shutil.rmtree(args.lr_dir_normalize)
        os.makedirs(args.lr_dir_normalize)
        if os.path.exists(args.gt_dir_normalize):
            shutil.rmtree(args.gt_dir_normalize)
        os.makedirs(args.gt_dir_normalize)
    except Exception as e:
        print(f"Error creating LR directory: {e}")
        return

    gt_flow_file_names = os.listdir(args.gt_dir)
    if not gt_flow_file_names:
        print("No gt flow files found in the GT directory.")
        return

    if args.all_flow_conditions_used:
        # Use a dictionary to store a list of filenames for each group
        groups = defaultdict(list)

        # Group file names by the number following "PIV-"
        for file in gt_flow_file_names:
            group_num = re.findall(r'\d+', file)[0]
            groups[group_num].append(file)

        # Sorts the list of filenames for each group and
        # merges the sorted list into an ordered list
        picked_flow_files = []
        for group_num in sorted(groups.keys(), key=int):
            sorted_group_files = sorted(groups[group_num], key=lambda x: int(re.findall(r'\d+', x)[1]))
            picked_flow_files += sorted_group_files

    else:
        # Filter out filenames starting with PIV-1
        piv1_files = [file for file in gt_flow_file_names if file.startswith("PIV-1")]
        # Extracts the numbers in the filenames and sorts them in order from smallest to largest
        picked_flow_files = sorted(piv1_files, key=lambda x: int(re.findall(r'\d+', x)[1]))

    # Splitting flow field with multiple threads
    progress_bar = tqdm(total=len(picked_flow_files), unit="flow", desc="Prepare split flow flied")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for flow_file_name in picked_flow_files:
        workers_pool.apply_async(worker, args=(flow_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(flow_file_name, args) -> None:
    global lr_flow
    try:
        flow_path = os.path.join(args.gt_dir, flow_file_name)
        gt_flow = np.load(flow_path, allow_pickle=False)

        # Load the maximum and minimum values saved locally.
        flow_max_final = np.loadtxt("data\\minimax\\flow_max_final.txt")
        flow_min_final = np.loadtxt("data\\minimax\\flow_min_final.txt")
        flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
        flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]

        gt_flow = (gt_flow - flow_min_final) / (flow_max_final - flow_min_final)
        if not (np.all(0 <= gt_flow) and np.all(gt_flow <= 1)):
            print("Some elements of gt_flow are out of the range [0, 1].")

        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        if args.down_sampling_mode == 'random':
            pooling_choice = ['pool', 'gaussian', 'cubic', ]
            pooling_method = random.choice(pooling_choice)

            if pooling_method == 'pool':
                lr_flow = imgproc.poolingOverlap3D(arr3d=gt_flow,
                                                   ksize=(args.down_sampling_factor, args.down_sampling_factor),
                                                   stride=None,
                                                   method='mean',
                                                   pad=False)
            elif pooling_method == 'gaussian':
                gt_flow_tensor = torch.from_numpy(gt_flow).to(device)
                lr_flow_tensor = core.imresize(gt_flow_tensor,
                                               scale=1 / args.down_sampling_factor,
                                               antialiasing=True,
                                               kernel='gaussian', sigma=1)
                lr_flow = lr_flow_tensor.cpu().numpy()
            elif pooling_method == 'cubic':
                gt_flow_tensor = torch.from_numpy(gt_flow).to(device)
                lr_flow_tensor = core.imresize(gt_flow_tensor,
                                               scale=1 / args.down_sampling_factor,
                                               antialiasing=True,
                                               kernel='cubic')
                lr_flow = lr_flow_tensor.cpu().numpy()

        elif args.down_sampling_mode == 'pooling':
            lr_flow = imgproc.poolingOverlap3D(arr3d=gt_flow,
                                               ksize=(args.down_sampling_factor, args.down_sampling_factor),
                                               stride=None,
                                               method='mean',
                                               pad=False)
        elif args.down_sampling_mode == 'resize_cubic':
            gt_flow_tensor = torch.from_numpy(gt_flow).to(device)
            lr_flow_tensor = core.imresize(gt_flow_tensor,
                                           scale=1 / args.down_sampling_factor,
                                           antialiasing=True,
                                           kernel='cubic')
            lr_flow = lr_flow_tensor.cpu().numpy()

        elif args.down_sampling_mode == 'resize_gaussian':
            gt_flow_tensor = torch.from_numpy(gt_flow).to(device)
            lr_flow_tensor = core.imresize(gt_flow_tensor,
                                           scale=1 / args.down_sampling_factor,
                                           antialiasing=True,
                                           kernel='gaussian', sigma=1)
            lr_flow = lr_flow_tensor.cpu().numpy()

        else:
            raise ValueError("Invalid random_method. Options are 'random', 'resize' and 'pooling'.")

        if args.normalized_choice == 'normalized_used':
            # Check that each element is in the range of 0 to 1
            if not (np.all(0 <= lr_flow) and np.all(lr_flow <= 1)):
                print("Some elements of lr_flow are out of the range [0, 1].")
        elif args.normalized_choice == 'normalized_unused':
            pass
        else:
            raise ValueError("Invalid normalized_choice. Options are 'normalized_used' and 'normalized_unused'.")

        lr_flow_consecutive = np.ascontiguousarray(lr_flow)

        # Save flow field file, Continue to use the original file name
        output_file_path_lr = os.path.join(args.lr_dir_normalize, flow_file_name).__str__()
        np.save(output_file_path_lr, lr_flow_consecutive)

        output_file_path_gt = os.path.join(args.gt_dir_normalize, flow_file_name).__str__()
        if args.normalized_choice == 'normalized_used':
            np.save(output_file_path_gt, gt_flow)
        elif args.normalized_choice == 'normalized_unused':
            gt_flow_value = gt_flow * (flow_max_final - flow_min_final) + flow_min_final
            np.save(output_file_path_gt, gt_flow_value)
        else:
            raise ValueError("Invalid normalized_choice. Options are 'normalized_used' and 'normalized_unused'.")

    except Exception as e:
        print(f"Error processing {flow_file_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paired GT and LR data pairs.")
    parser.add_argument("--gt_dir", type=str,
                        help="Path to input GT flow field distribution directory.")
    parser.add_argument("--lr_dir_normalize", type=str,
                        help="Path to save generator LR Normalized flow field distribution directory.")
    parser.add_argument("--gt_dir_normalize", type=str,
                        help="Path to save generator GT Normalized flow field distribution directory.")

    parser.add_argument("--down_sampling_mode", type=str,
                        help="Which down sampling method should be used.")

    parser.add_argument("--normalized_choice", type=str,
                        help="Whether it is normalized.", default="normalized_used")

    parser.add_argument("--down_sampling_factor", type=int,
                        help="Scaling range of the downs ample.")

    parser.add_argument("--all_flow_conditions_used", action="store_true",
                        help="If this flag is present, data from all flow conditions will be used. If absent, "
                             "it's considered False.")
    parser.set_defaults(all_flow_conditions_used=False)

    # parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")

    args = parser.parse_args()

    main(args)
