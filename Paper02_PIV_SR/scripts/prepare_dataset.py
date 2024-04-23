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
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import multiprocessing


def main(args) -> None:
    try:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    flow_file_names = os.listdir(args.dataset_dir)
    if not flow_file_names:
        print("No flow files found in the dataset directory.")
        return

    if args.all_flow_conditions_used:
        # Use a dictionary to store a list of filenames for each group
        groups = defaultdict(list)

        # Group file names by the number following "PIV-"
        for file in flow_file_names:
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
        piv1_files = [file for file in flow_file_names if file.startswith("PIV-1")]
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
    try:
        flow_path = os.path.join(args.dataset_dir, flow_file_name)
        flow = np.load(flow_path, allow_pickle=False)

        flow_height, flow_width = flow.shape[1:3]

        # Just need to find the top and left coordinates of the flow field
        top = random.randint(0, flow_height - args.flow_crop_height)
        left = random.randint(0, flow_width - args.flow_crop_width)
        # print(f"top: {top}, left: {left}")

        # Crop flow field patch
        crop_flow = flow[:, top:top + args.flow_crop_height, left:left + args.flow_crop_width]
        crop_flow = np.ascontiguousarray(crop_flow)

        # Save flow field file, Continue to use the original file name
        output_file_path = os.path.join(args.output_dir, flow_file_name).__str__()
        np.save(output_file_path, crop_flow)

    except Exception as e:
        print(f"Error processing {flow_file_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--dataset_dir", type=str,
                        help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str,
                        help="Path to generator image directory.")

    parser.add_argument("--flow_crop_height", type=int,
                        help="Low-resolution raw Flow field distribution height.")
    parser.add_argument("--flow_crop_width", type=int,
                        help="Low-resolution raw Flow field distribution width.")

    parser.add_argument("--all_flow_conditions_used", action="store_true",
                        help="If this flag is present, data from all flow conditions will be used. If absent, "
                             "it's considered False.")
    parser.set_defaults(all_flow_conditions_used=False)

    # parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")

    args = parser.parse_args()

    main(args)