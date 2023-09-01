"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The link to the reference code repository is as follows:
    https://github.com/guilindner/VortexFitting
"""

import numpy as np
import os
import argparse


def ASCII2D_Convert_matrix(ASCII2D, matrix_H, matrix_W, index):
    """

    Args:
        ASCII2D: the original 2D ASCII file
        matrix_H: the number of rows of the matrix
        matrix_W: the number of columns of the matrix
        index: the index of the column to be extracted

    Returns: the matrix of the specified column

    """
    matrix = np.empty((matrix_H, matrix_W), dtype=np.float32)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            matrix[y, x] = ASCII2D[y * matrix.shape[1] + x, index]
    return matrix


def Vortex_Identification(args):
    """

    Args:
        args: Enter the set combination of calculation parameters
    """
    result_files = os.listdir(args.re_results_folder)
    file_sorted = sorted(result_files, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]), reverse=False)

    Progress_bar = 1
    file_nums = int(len(file_sorted))
    print('The number of files that need to be extracted is: {}'.format(file_nums))

    for file in file_sorted:
        file_path = args.re_results_folder + '/' + file
        if args.file_type == 'txt':
            re_result_sheet = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
        elif args.file_type == 'dat':
            re_result_sheet = np.genfromtxt(fname=file_path,
                                            skip_header=6,
                                            skip_footer=-1,
                                            names=["x", "y", "u", "v", "isNaN"],
                                            dtype=np.float32,
                                            delimiter=' ')
            re_result_sheet = np.array(re_result_sheet.tolist(), dtype=np.float32)
        else:
            raise ValueError('The file type is not supported, please check the file type.')

        num_tag = os.path.splitext(file.split('_')[1])[0]

        index_x, index_y, index_u, index_v, index_d, index_M = 0, 1, 2, 3, 4, 5

        # Get the grid number of coordinates X and Y
        dx_tmp = np.array(re_result_sheet[:, index_x])
        for i in range(1, dx_tmp.shape[0]):
            if dx_tmp[i] == dx_tmp[0]:
                x_coordinate_size = i
                y_coordinate_size = int(dx_tmp.shape[0] / x_coordinate_size)
                break

        if y_coordinate_size * x_coordinate_size == dx_tmp.shape[0]:
            u_velocity_matrix = ASCII2D_Convert_matrix(ASCII2D=re_result_sheet,
                                                       matrix_H=y_coordinate_size,
                                                       matrix_W=x_coordinate_size,
                                                       index=index_u)
            v_velocity_matrix = ASCII2D_Convert_matrix(ASCII2D=re_result_sheet,
                                                       matrix_H=y_coordinate_size,
                                                       matrix_W=x_coordinate_size,
                                                       index=index_v)

            tmp_x = ASCII2D_Convert_matrix(ASCII2D=re_result_sheet,
                                           matrix_H=y_coordinate_size,
                                           matrix_W=x_coordinate_size,
                                           index=index_x)
            tmp_y = ASCII2D_Convert_matrix(ASCII2D=re_result_sheet,
                                           matrix_H=y_coordinate_size,
                                           matrix_W=x_coordinate_size,
                                           index=index_y)

            x_coordinate_matrix = tmp_x[1, :]
            y_coordinate_matrix = tmp_y[:, 1]

            x_coordinate_step = round((np.max(x_coordinate_matrix) - np.min(x_coordinate_matrix)) /
                                      (np.size(x_coordinate_matrix) - 1), 32)
            y_coordinate_step = round((np.max(y_coordinate_matrix) - np.min(y_coordinate_matrix)) /
                                      (np.size(y_coordinate_matrix) - 1), 32)

            dudx = np.zeros_like(u_velocity_matrix).astype(np.float32)
            dudy = np.zeros_like(u_velocity_matrix).astype(np.float32)
            dvdx = np.zeros_like(u_velocity_matrix).astype(np.float32)
            dvdy = np.zeros_like(u_velocity_matrix).astype(np.float32)

            if args.finite_difference_scheme == '2':  # 'second_order_diff'
                dx = x_coordinate_step  # only for homogeneous mesh
                dy = y_coordinate_step  # only for homogeneous mesh

                dudy, dudx = np.gradient(u_velocity_matrix, dy, dx)
                dvdy, dvdx = np.gradient(v_velocity_matrix, dy, dx)

            if args.finite_difference_scheme == '4':  # 'fourth_order_diff'
                dx = x_coordinate_step  # only for homogeneous mesh
                dy = y_coordinate_step  # only for homogeneous mesh

                # 4th order central difference
                # Part of the fourth-order difference that cannot be processed will not be processed
                dudx[:, 2:-2] = (u_velocity_matrix[:, 0:-4] -
                                 8 * u_velocity_matrix[:, 1:-3] +
                                 8 * u_velocity_matrix[:, 3:-1] -
                                 u_velocity_matrix[:, 4:]) / (12 * dx)
                dudy[2:-2, :] = (u_velocity_matrix[0:-4, :] -
                                 8 * u_velocity_matrix[1:-3, :] +
                                 8 * u_velocity_matrix[3:-1, :] -
                                 u_velocity_matrix[4:, :]) / (12 * dy)
                dvdx[:, 2:-2] = (v_velocity_matrix[:, 0:-4] -
                                 8 * v_velocity_matrix[:, 1:-3] +
                                 8 * v_velocity_matrix[:, 3:-1] -
                                 v_velocity_matrix[:, 4:]) / (12 * dx)
                dvdy[2:-2, :] = (v_velocity_matrix[0:-4, :] -
                                 8 * v_velocity_matrix[1:-3, :] +
                                 8 * v_velocity_matrix[3:-1, :] -
                                 v_velocity_matrix[4:, :]) / (12 * dy)

            vorticity = dvdx - dudy

            if args.detection_method == 'Q':
                if Progress_bar == 1:
                    print('Detection method: Q criterion')
                q_matrix = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                for i in range(u_velocity_matrix.shape[0]):
                    for j in range(u_velocity_matrix.shape[1]):
                        q_matrix[i, j] = -0.5 * (dudx[i, j] ** 2 + dvdy[i, j] ** 2) - dudy[i, j] * dvdx[i, j]
                detection_field = q_matrix
            elif args.detection_method == 'Delta':
                if Progress_bar == 1:
                    print('Detection method: Delta criterion')
                q_matrix = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                r_matrix = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                delta = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                for i in range(u_velocity_matrix.shape[0]):
                    for j in range(u_velocity_matrix.shape[1]):
                        q_matrix[i, j] = -0.5 * (dudx[i, j] ** 2 + dvdy[i, j] ** 2) - dudy[i, j] * dvdx[i, j]
                        r_matrix[i, j] = dudx[i, j] * dvdy[i, j] - dvdx[i, j] * dudy[i, j]  # determinant of Velocity tensor
                        delta[i, j] = (q_matrix[i, j] / 3) ** 3 + (r_matrix[i, j] / 2) ** 2  # Incompressible flow
                detection_field = delta
            elif args.detection_method == 'Omega':
                if Progress_bar == 1:
                    print('Detection method: Omega criterion from Liu')
                # D = S + R
                # The symmetry tensor of the velocity tensor
                S_matrix = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                # The inverse symmetry tensor of the velocity tensor
                R_matrix = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                omega = np.zeros((u_velocity_matrix.shape[0], u_velocity_matrix.shape[1]))
                for i in range(u_velocity_matrix.shape[0]):
                    for j in range(u_velocity_matrix.shape[1]):
                        S_matrix[i, j] = dudx[i, j] ** 2 + dvdy[i, j] ** 2 + 0.5 * ((dudy[i, j] + dvdx[i, j]) ** 2)
                        R_matrix[i, j] = 0.5 * ((dvdx[i, j] - dudy[i, j]) ** 2)
                        omega[i, j] = R_matrix[i, j] / (S_matrix[i, j] + R_matrix[i, j] + 10e-64)
                detection_field = omega
            else:
                raise ValueError('The detection method is not supported')

            np.savetxt(fname=args.vortex_identification_folder + '/' + 'VortexField' + args.detection_method + args.file_type + '_' + num_tag + '.txt',
                       X=detection_field, fmt='%.32f')

            print('Vortex Testing Progress {}/{}'.format(Progress_bar, file_nums))
            Progress_bar += 1

        else:
            raise ValueError('The grid number cannot be divisible when obtaining coordinates X and Y')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', "--re_results_folder", type=str, help='Input file')

    parser.add_argument('-o', "--vortex_identification_folder", type=str,
                        help='To specify an output directory')

    parser.add_argument('-s', "--finite_difference_scheme", default=2, type=str,
                        help='Scheme for differencing\n'
                             '2 = second order \n'
                             '22 = least-square filter (default)\n'
                             '4 = fourth order\n'
                             'second difference is enough')

    parser.add_argument('-d', "--detection_method", default='Omega', type=str,
                        help='Detection method:\n'
                             'Q = Q criterion\n'
                             'delta = delta criterion\n'
                             'swirling = 2D Swirling Strength (default)\n'
                             'omega = Omega criterion from Liu')

    parser.add_argument('-f', "--file_type", default='txt', type=str,
                        help='Type of data file:\n'
                             '.txt file\n'
                             '.dat file')

    args = parser.parse_args()

    Vortex_Identification(args)
