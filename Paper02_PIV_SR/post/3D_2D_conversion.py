"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
# Convert three-dimensional inference .npy data calculated by neural network reasoning into two-dimensional .txt data

import numpy as np
import os
import shutil
from natsort import natsorted
import argparse
from tqdm import tqdm


def extract_data(args):
    """

    Args:
        args:
    """
    # npy_file_names = os.listdir(args.folder_path_input)
    # List all files in the input directory
    all_file_names = os.listdir(args.folder_path_input)
    # Filter to include only .npy files
    npy_file_names = [file for file in all_file_names if file.endswith('.npy')]
    npy_file_names_sorted = natsorted(npy_file_names)

    if os.path.exists(args.folder_path_output):
        shutil.rmtree(args.folder_path_output)
    os.makedirs(args.folder_path_output, mode=0o755)

    files_count = int(len(npy_file_names_sorted))
    print('The number of flow field data files that need to be dimensionally transformed is {}'.format(files_count))

    # Load the maximum and minimum values saved locally.
    flow_max_final = np.loadtxt("data\\minimax\\flow_max_final.txt")
    flow_min_final = np.loadtxt("data\\minimax\\flow_min_final.txt")
    flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
    flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]

    for file_name in tqdm(npy_file_names_sorted, desc='Data Dimensional Transformation', unit='files', ):
        file_path = os.path.join(args.folder_path_input, file_name)
        data_3d = np.load(file_path, allow_pickle=True).astype(np.float32)

        C = data_3d.shape[0]
        if args.if_normalized_velocity == 'V_nor':
            data_3d = data_3d[0:C, :, :] * (flow_max_final[0:C, :, :] - flow_min_final[0:C, :, :]) + flow_min_final[0:C, :, :]
        else:
            pass

        # Assuming 'data' is your 3D array
        output_2d = np.zeros((data_3d.shape[1] * data_3d.shape[2], data_3d.shape[0]), dtype=np.float64)

        # data, ["u", "v", "vector_direction"], 3D Data Format Conversion 2D Data Format
        for c in range(data_3d.shape[0]):
            for y in range(data_3d.shape[1]):
                for x in range(data_3d.shape[2]):
                    output_2d[y * data_3d.shape[2] + x, c] = data_3d[c, data_3d.shape[1] - y - 1, x]

        if args.if_velocity_calculation == "V_True":
            # Calculate the magnitude of the velocity vector
            v_mag = np.empty((output_2d.shape[0], 1))
            for i in range(output_2d.shape[0]):
                v_mag[i] = np.sqrt(output_2d[i, 0] ** 2 + output_2d[i, 1] ** 2)
            output_2d = np.hstack((output_2d, v_mag))
        else:
            pass

        # Define the size of the grid and the scale of the axes
        grid_shape = (data_3d.shape[1], data_3d.shape[2])  # (236, 116)

        # Dimensions along X and Y
        length_x = 0.05*20/22  # m
        length_y = (0.05*20/22)*44/20  # m

        # hydraulic diameter
        a = 0.003
        b = 0.05
        D = 4 * a * b / (2 * (a + b))

        x = np.linspace(0, length_x/D, grid_shape[1])
        y = np.linspace(0, length_y/D, grid_shape[0])
        x_mesh, y_mesh = np.meshgrid(x, y)

        # # Create an empty numpy array to store the coordinates XY
        # coords = np.empty((grid_shape[0] * grid_shape[1], 2))

        # Flatten the mesh grids and pair them as coordinates
        coords = np.vstack((x_mesh.ravel(), y_mesh.ravel())).T

        # Concatenating array coords and array output_2d
        data_reconstruction = np.hstack((coords, output_2d))

        # Split filename and file extension of file_name
        name_without_extension, extension = os.path.splitext(file_name)
        file_name_txt = name_without_extension + '.txt'

        # save results as txt
        data_save_path = os.path.join(args.folder_path_output, file_name_txt)
        np.savetxt(data_save_path, data_reconstruction, fmt='%.32f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Inference Data Format Conversion 2D Data Format.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the 3D data .dat file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of the 2D data .dat file.")
    parser.add_argument("--if_velocity_calculation", type=str, help="Whether to calculate the velocity magnitude.",
                        default="V_True")
    parser.add_argument("--if_normalized_velocity", type=str, help="Input is not the normalized speed",
                        default="V_nor")

    args = parser.parse_args()

    extract_data(args)


