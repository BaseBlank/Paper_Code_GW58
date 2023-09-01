"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
# Convert the three-dimensional tset data calculated by neural network reasoning into two-dimensional data

import numpy as np
import os
import argparse


def extract_data(args):
    """

    Args:
        args:
    """
    dat_file_names_GT = os.listdir(args.folder_path_input_GT)
    dat_file_names_GT_sorted = sorted(dat_file_names_GT, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                      reverse=False)

    progression = 1
    File_num = int(len(dat_file_names_GT_sorted))
    print('The number of flow field data files that need to be dimensionally transformed is {}'.format(File_num))

    for dat_file_name in dat_file_names_GT_sorted:
        file_path = args.folder_path_input_GT + '/' + dat_file_name
        data_3D_GT = np.load(file_path).astype(np.float32)

        Num_tag = os.path.splitext(dat_file_name.split('_')[1])[0]

        # data, ["u", "v", "vector_direction"], 3D Data Format Conversion 2D Data Format
        data_GT = np.zeros((data_3D_GT.shape[0] * data_3D_GT.shape[1], 3), dtype=np.float32)
        for c in range(3):
            for y in range(data_3D_GT.shape[0]):
                for x in range(data_3D_GT.shape[1]):
                    data_GT[y * data_3D_GT.shape[1] + x, c] = data_3D_GT[data_3D_GT.shape[0] - y - 1, x, c]

        # Define the size of the grid and the scale of the axes
        grid_size_GT = (data_3D_GT.shape[0], data_3D_GT.shape[1])  # (56, 28)
        scale_GT = float(args.grid_scale)  # Calculated from the PIV export data, 0.0016343949044585

        # Create an empty numpy array to store the GT coordinates
        coords_GT = np.empty((grid_size_GT[0] * grid_size_GT[1], 2))

        # Fill the numpy array with the coordinates of each point
        for i in range(grid_size_GT[0]):
            for j in range(grid_size_GT[1]):
                x = j * scale_GT + float(args.Array2D_X_start)  # 0.003829670912951
                y = i * scale_GT + float(args.Array2D_Y_start)  # 0.005141985138004
                # coords[i*grid_size[0]+j] = [x, y]
                coords_GT[i * grid_size_GT[1] + j] = [x, y]

        # data_add, ["x", "y", "u", "v", "vector_direction", "vector_magnitude"], 2D Data Format
        data_add = np.zeros((data_GT.shape[0], data_GT.shape[1] + 1 + 2), dtype=np.float32)
        for xy in range(data_GT.shape[0]):
            data_add[xy, 0:2] = coords_GT[xy, 0:2]
        for uvd in range(data_GT.shape[0]):
            data_add[uvd, 2:5] = data_GT[uvd, 0:3]
        for d in range(data_GT.shape[0]):
            if data_add[d, 4] > 360:
                data_add[d, 4] = 360
        for M in range(data_GT.shape[0]):
            data_add[M, 5] = np.sqrt(data_add[M, 2] ** 2 + data_add[M, 3] ** 2)

        np.savetxt(fname=args.folder_path_output + '/' + args.variable_name + '_' + Num_tag + '.txt',
                   X=data_add, fmt='%.32f')

        print('维度转换进度 {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Test Data Format Conversion 2D Data Format.")
    parser.add_argument("--folder_path_input_GT", type=str, help="The folder location of the 3D GT data .dat file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of the 2D data .dat file.")
    parser.add_argument("--grid_scale", type=str, default=0.0016343949044585,
                        help="The scale of the XY coordinates grid change.")
    parser.add_argument("--Array2D_X_start", type=str, default=0.003829670912951,
                        help="The starting coordinate of the X coordinate of the two-dimensional array.")
    parser.add_argument("--Array2D_Y_start", type=str, default=0.005141985138004,
                        help="The starting coordinate of the Y coordinate of the two-dimensional array.")
    parser.add_argument("--variable_name", type=str, default='2DArrayGT',
                        help="The name of the 2D data variable to extract.")

    args = parser.parse_args()

    extract_data(args)
