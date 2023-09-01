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
import random
import shutil

from tqdm import tqdm

random.seed(928)  # 只有效一次


def main(args) -> None:
    if not os.path.exists(args.valid_images_dir):
        os.makedirs(args.valid_images_dir)

    train_files = os.listdir(f"{args.train_images_dir}")
    valid_files = random.sample(train_files, int(len(train_files) * args.valid_samples_ratio))

    process_bar = tqdm(valid_files, total=len(valid_files), unit="image", desc="Split train/valid dataset")

    for image_file_name in process_bar:
        shutil.copyfile(f"{args.train_images_dir}/{image_file_name}", f"{args.valid_images_dir}/{image_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts.")
    parser.add_argument("--train_images_dir", type=str, help="Path to train image directory.")
    parser.add_argument("--valid_images_dir", type=str, help="Path to valid image directory.")
    parser.add_argument("--valid_samples_ratio", type=float, help="What percentage of the data is extracted from the train set into the valid set.")
    args = parser.parse_args()

    main(args)
