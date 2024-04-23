"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
import numpy as np
import os
import argparse


# 调用上一级目录的imgproc.py程序的中的image_resize函数
import sys
sys.path.append('F:\\Code\\RDN')
# sys模块打印当前的系统目录
print(sys.path)
from imgproc import imresize


# 调用上一级目录的config.py程序的中的upscale_factor参数
import config

print(config.upscale_factor)


def extract_data(args):
    """

    Args:
        args:
    """
    dat_file_names = os.listdir(args.folder_path_input)
    dat_file_names_sorted = sorted(dat_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

    progression = 1
    File_num = int(len(dat_file_names_sorted))
    print('The number of test data files is {}'.format(File_num))

    for dat_file_name in dat_file_names_sorted:
        file_path = args.folder_path_input + '/' + dat_file_name
        data_3D_original = np.load(file_path).astype(np.float32)
        data_3D_cropped = data_3D_original[1:57, 1:29, :]  # shape = (56, 28, 3)
        data_3D_lr = imresize(data_3D_cropped, 1 / config.upscale_factor)  # [H,W,C]

        Num_tag = os.path.splitext(dat_file_name.split('_')[1])[0]

        np.save(args.folder_path_output_gt + '/' + args.variable_name_gt + '_' + Num_tag + '.npy', data_3D_cropped)
        np.save(args.folder_path_output_lr + '/' + args.variable_name_lr + '_' + Num_tag + '.npy', data_3D_lr)

        print('Build progress {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test data generation for validating the algorithm's superiority over interpolation.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the original data set .dat file.")
    parser.add_argument("--folder_path_output_gt", type=str,
                        help="The folder location of cropped dataset for scaling.")
    parser.add_argument("--folder_path_output_lr", type=str,
                        help="The folder location of down sampling dataset for scaling.")
    parser.add_argument("--variable_name_gt", type=str, default="GT", help="The variable name of cropped dataset.")
    parser.add_argument("--variable_name_lr", type=str, default="LR", help="The variable name of down sampling dataset.")

    args = parser.parse_args()

    extract_data(args)
