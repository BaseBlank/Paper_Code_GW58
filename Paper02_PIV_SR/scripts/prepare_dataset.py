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
import multiprocessing
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def main(args) -> None:
    """

    Args:
        args:
    """
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Get all image paths
    image_file_names = os.listdir(args.images_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split image")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    """

    Args:
        image_file_name:
        args:
    """
    image = cv2.imread(f"{args.images_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[0:2]

    index = 1
    if image_height >= args.image_size and image_width >= args.image_size:
        for pos_y in range(0, image_height - args.image_size + 1, args.step):
            for pos_x in range(0, image_width - args.image_size + 1, args.step):
                # Crop
                crop_image = image[pos_y: pos_y + args.image_size, pos_x:pos_x + args.image_size, ...]
                crop_image = np.ascontiguousarray(crop_image)
                # Save image
                cv2.imwrite(f"{args.output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}", crop_image)

                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to generator image directory.")
    parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.")
    parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    args = parser.parse_args()

    main(args)
