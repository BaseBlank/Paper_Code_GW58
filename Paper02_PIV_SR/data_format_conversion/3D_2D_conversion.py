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
# Convert the three-dimensional inference data calculated by neural network reasoning into two-dimensional data

import numpy as np
import os
import argparse


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
    print('The number of flow field data files that need to be dimensionally transformed is {}'.format(File_num))

    for dat_file_name in dat_file_names_sorted:
        file_path = args.folder_path_input + '/' + dat_file_name
        data_3D_orign = np.load(file_path).astype(np.float32)

        if args.uv_convert_sign:
            data_3D = data_3D_orign
            for i in range(data_3D_orign.shape[0]):
                for j in range(data_3D_orign.shape[1]):
                    if 0 < data_3D_orign[i, j, 2] <= 90:
                        pass
                    elif 90 < data_3D_orign[i, j, 2] <= 180:
                        data_3D[i, j, 0] = -data_3D_orign[i, j, 0]
                    elif 180 < data_3D_orign[i, j, 2] <= 270:
                        data_3D[i, j, 0] = -data_3D_orign[i, j, 0]
                        data_3D[i, j, 1] = -data_3D_orign[i, j, 1]
                    else:
                        data_3D[i, j, 1] = -data_3D_orign[i, j, 1]
        else:
            data_3D = data_3D_orign

        # data, ["u", "v", "vector_direction"], 3D Data Format Conversion 2D Data Format
        data = np.zeros((data_3D.shape[0] * data_3D.shape[1], 3), dtype=np.float32)
        for c in range(3):
            for y in range(data_3D.shape[0]):
                for x in range(data_3D.shape[1]):
                    data[y * data_3D.shape[1] + x, c] = data_3D[data_3D.shape[0] - y - 1, x, c]

        # Define the size of the grid and the scale of the axes
        grid_size = (data_3D.shape[0], data_3D.shape[1])  # (236, 116)
        scale = 0.0016343949044585*59/236  # Calculated from the PIV export data

        # Create an empty numpy array to store the coordinates
        coords = np.empty((grid_size[0] * grid_size[1], 2))

        # Fill the numpy array with the coordinates of each point
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = j * scale + 0.001634394904458 / 8 + 0.001378078556264
                y = i * scale + 0.001634394904458 / 8 + 0.001055997876858
                # coords[i*grid_size[0]+j] = [x, y]
                coords[i * grid_size[1] + j] = [x, y]

        # data_add, ["x", "y", "u", "v", "vector_direction", "vector_magnitude"], 2D Data Format
        data_add = np.zeros((data.shape[0], data.shape[1]+1+2), dtype=np.float32)
        for xy in range(data.shape[0]):
            data_add[xy, 0:2] = coords[xy, 0:2]
        for uvd in range(data.shape[0]):
            data_add[uvd, 2:5] = data[uvd, 0:3]
        for M in range(data.shape[0]):
            data_add[M, 5] = np.sqrt(data_add[M, 2] ** 2 + data_add[M, 3] ** 2)
        for d in range(data.shape[0]):
            if data_add[d, 4] > 360:
                data_add[d, 4] = 360

        if progression < 10:
            Num_tag = '0000' + str(progression)
        elif 10 <= progression < 100:
            Num_tag = '000' + str(progression)
        elif 100 <= progression < 1000:
            Num_tag = '00' + str(progression)
        elif 1000 <= progression < 10000:
            Num_tag = '0' + str(progression)
        else:
            Num_tag = str(progression)

        np.savetxt(fname=args.folder_path_output + '/' + args.variable_name + '_' + Num_tag + '.txt',
                   X=data_add, fmt='%.32f')

        print('维度转换进度 {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Inference Data Format Conversion 2D Data Format.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the 3D data .dat file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of the 2D data .dat file.")
    parser.add_argument("--variable_name", type=str, default='2DArray',
                        help="The name of the 2D data variable to extract.")

    parser.add_argument("--uv_convert_sign", type=bool, default=False,
                        help="Whether to convert the sign of uv speed.")

    args = parser.parse_args()

    extract_data(args)

