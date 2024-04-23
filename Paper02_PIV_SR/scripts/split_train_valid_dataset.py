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
from tqdm import tqdm
import random

random.seed(928)  # Only work once


def create_directory_structure(args) -> None:
    """
    Create the desired directory structure,
    including the training, validation, and test directories.
    """
    directories = [args.train_dataset_dir, args.valid_dataset_dir, args.test_dataset_dir]
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, mode=0o755)  # Set the directory permission explicitly


def main(args) -> None:
    create_directory_structure(args)

    flow_files = os.listdir(f"{args.raw_dataset_dir}")
    if args.all_flow_conditions_used:
        picked_flow_files = flow_files
    else:
        # Filter out filenames starting with PIV-1
        picked_flow_files = [file for file in flow_files if file.startswith("PIV-1")]

    total_files_count = len(picked_flow_files)

    # Calculate the number of files for validation and test sets
    valid_files_count = int(total_files_count * args.valid_samples_ratio)
    test_files_count = int(total_files_count * args.test_samples_ratio)

    # Get the validation and test files using random.sample
    valid_files = random.sample(picked_flow_files, valid_files_count)
    remaining_files = [file for file in picked_flow_files if file not in valid_files]
    test_files = random.sample(remaining_files, test_files_count)

    # The training files are the remaining files after excluding validation and test files
    train_files = [file for file in remaining_files if file not in test_files]

    assert len(train_files) + len(valid_files) + len(test_files) == total_files_count, \
        "The number of files in the train, valid and test sets does not match the total number of files!"

    # Print the number of files in each set
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(valid_files)} files")
    print(f"Test set: {len(test_files)} files")

    process_bar_train = tqdm(train_files, total=len(train_files), unit="train files", desc="Split train dataset")
    for train_file_name in process_bar_train:
        shutil.copyfile(f"{args.raw_dataset_dir}/{train_file_name}", f"{args.train_dataset_dir}/{train_file_name}")
        process_bar_train.update(1)
    process_bar_train.close()

    process_bar_valid = tqdm(valid_files, total=len(valid_files), unit="valid files", desc="Split valid dataset")
    for valid_file_name in process_bar_valid:
        shutil.copyfile(f"{args.raw_dataset_dir}/{valid_file_name}", f"{args.valid_dataset_dir}/{valid_file_name}")
        process_bar_valid.update(1)
    process_bar_valid.close()

    process_bar_test = tqdm(test_files, total=len(test_files), unit="test files", desc="Split test dataset")
    for test_file_name in process_bar_test:
        shutil.copyfile(f"{args.raw_dataset_dir}/{test_file_name}", f"{args.test_dataset_dir}/{test_file_name}")
        process_bar_test.update(1)
    process_bar_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train, valid and test dataset.")
    parser.add_argument("--raw_dataset_dir", type=str,
                        help="Path to train raw flow dataset directory.")
    parser.add_argument("--train_dataset_dir", type=str,
                        help="Path to flow train dataset directory.")
    parser.add_argument("--valid_dataset_dir", type=str,
                        help="Path to flow valid dataset directory.")
    parser.add_argument("--test_dataset_dir", type=str,
                        help="Path to flow test dataset directory.")

    parser.add_argument("--valid_samples_ratio", type=float,
                        help="Percentage of extracts from the original dataset to the validation dataset.")
    parser.add_argument("--test_samples_ratio", type=float,
                        help="Percentage of extracts from the original dataset to the test dataset.")

    parser.add_argument("--all_flow_conditions_used", action="store_true",
                        help="If this flag is present, data from all flow conditions will be used. If absent, "
                             "it's considered False.")
    parser.set_defaults(all_flow_conditions_used=False)

    args = parser.parse_args()

    main(args)
