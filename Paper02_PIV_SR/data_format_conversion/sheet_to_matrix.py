"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================

# Convert the worksheet txt file back to the form of a two-dimensional matrix for easy drawing

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
    print('The number of files that need to be extracted is {}'.format(File_num))

    if args.extracted_variable == 'x':
        var_index = 0
    elif args.extracted_variable == 'y':
        var_index = 1
    elif args.extracted_variable == 'u':
        var_index = 2
    elif args.extracted_variable == 'v':
        var_index = 3
    elif args.extracted_variable == 'd':
        var_index = 4
    elif args.extracted_variable == 'm':
        var_index = 5
    else:
        raise ValueError('The extracted variable is not in the XY arrangement .txt file')

    for file_name in dat_file_names_sorted:
        file_path = args.folder_path_input + '/' + file_name
        if args.file_type == 'txt':
            data_sheet = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
        elif args.file_type == 'dat':
            data_sheet = np.genfromtxt(fname=file_path,
                                       skip_header=6,
                                       skip_footer=-1,
                                       names=["x", "y", "u", "v", "isNaN"],
                                       dtype=np.float32,
                                       delimiter=' ')
            data_sheet = np.array(data_sheet.tolist(), dtype=np.float32)
        else:
            raise ValueError('The file type is not supported, please check the file type.')

        Num_tag = os.path.splitext(file_name.split('_')[1])[0]

        # origin的云图根据矩阵绘制，矩阵的坐标原点在左上角，而不是真实坐标的左下角
        data = np.zeros((int(args.matrix_rows_num_Y), int(args.matrix_cols_num_X)), dtype=np.float32)
        if data.shape[0] * data.shape[1] == data_sheet.shape[0]:
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    data[data.shape[0] - y - 1, x] = data_sheet[y * data.shape[1] + x, var_index]
        else:
            raise ValueError('The number of data in the matrix does not match the number of data in the sheet')

        np.savetxt(
            fname=args.folder_path_output + '/' + args.variable_name + args.extracted_variable + '_' + Num_tag + '.txt',
            X=data, fmt='%.32f')

        print('Data extraction progress {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the worksheet .txt file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of save matrix .txt file.")
    parser.add_argument("--extracted_variable", type=str,
                        help="The variables that need to be extracted and converted to matrices.")
    parser.add_argument("--matrix_rows_num_Y", type=str, help="The number of rows of the matrix, i.e. the height.")
    parser.add_argument("--matrix_cols_num_X", type=str, help="The number of columns of the matrix, i.e. the width.")
    parser.add_argument("--variable_name", type=str, help="The name of the data variable to extract.")
    parser.add_argument("--file_type", default='txt', type=str, help='Type of data file:\n'
                                                                     '.txt file\n'
                                                                     '.dat file')

    args = parser.parse_args()

    extract_data(args)
