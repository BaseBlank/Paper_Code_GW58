"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================

# Traditional interpolation method realizes conversion of LR to HR data, [u,v,d], 3D -> [m], 2D
# Deprecated since version 1.10.0: interp2d is deprecated in SciPy 1.10 and will be removed in SciPy 1.12.0.
# Use RegularGridInterpolator instead.

import numpy as np
import cv2
import os
import argparse
import sys
sys.path.append('F:\\Code\\RDN')
from imgproc import image_resize
import config


def Interpolation_data(args):
    """

    Args:
        args:
    """
    dat_file_names = os.listdir(args.folder_path_input)
    dat_file_names_sorted = sorted(dat_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

    progression = 1
    File_num = int(len(dat_file_names_sorted))
    print('The number of data files to be interpolated is {}'.format(File_num))

    for file_name in dat_file_names_sorted:
        file_path = args.folder_path_input + '/' + file_name
        data_LR_3D = np.load(file_path).astype(np.float32)  # [u, v, d]

        Num_tag = os.path.splitext(file_name.split('_')[1])[0]

        if args.interpolation_method == 'bicubic':
            data_HR_3D = image_resize(data_LR_3D, config.upscale_factor)  # [H,W,C]
        elif args.interpolation_method == 'linear':
            data_HR_3D = cv2.resize(data_LR_3D,
                                    dsize=None,
                                    fx=config.upscale_factor,
                                    fy=config.upscale_factor,
                                    interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('The interpolation method is not supported.')

        data = np.zeros((data_HR_3D.shape[0], data_HR_3D.shape[1]), dtype=np.float32)

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                data[y, x] = np.sqrt(data_HR_3D[y, x, 0] ** 2 + data_HR_3D[y, x, 1] ** 2)

        np.savetxt(
            fname=args.folder_path_output + '/' + args.interpolation_method + args.variable_name + '_' + Num_tag + '.txt',
            X=data, fmt='%.32f')

        print('Data extraction progress {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the LR data file.")
    parser.add_argument("--folder_path_output", type=str,
                        help="The folder location of interpolation conversion HR data file.")
    parser.add_argument("--interpolation_method", type=str,
                        help="The interpolation method used, including {‘linear’, ‘bicubic’}.")
    parser.add_argument("--variable_name", default='m', type=str, help="The name of the data variable to extract.")

    args = parser.parse_args()

    Interpolation_data(args)
