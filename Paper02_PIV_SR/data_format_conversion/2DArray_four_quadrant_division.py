"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""

import numpy as np
import os

Array2D_path = "F:\\model_generator\\No_Heating\\center_blockage_0.7\\2DArray"
Array2D_file_names = os.listdir(Array2D_path)
Array2D_file_names_sorted = sorted(Array2D_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

sign_division_output_path = "F:\\model_generator\\No_Heating\\center_blockage_0.7\\2DArray_sign"

Progress_bar = 1
file_nums = int(len(Array2D_file_names_sorted))
print('The number of files that need to be extracted is: {}'.format(file_nums))

for Array2D_file_name in Array2D_file_names_sorted:
    Array2D_file_path = Array2D_path + "\\" + Array2D_file_name
    Array2D_data = np.loadtxt(Array2D_file_path, dtype=np.float32, delimiter=' ')

    Array2D_data_sign_division = np.zeros_like(Array2D_data, dtype=np.float32)

    Num_tag = os.path.splitext(Array2D_file_name.split('_')[1])[0]

    index_x, index_y, index_u, index_v, index_d, index_M = 0, 1, 2, 3, 4, 5

    for y in range(Array2D_data.shape[0]):
        if 0 <= Array2D_data[y, index_d] <= 90.0:
            Array2D_data_sign_division[y] = Array2D_data[y]
        elif 90 < Array2D_data[y, index_d] <= 180.0:
            Array2D_data_sign_division[y] = Array2D_data[y]
            Array2D_data_sign_division[y, index_u] = -Array2D_data[y, index_u]
        elif 180 < Array2D_data[y, index_d] <= 270.0:
            Array2D_data_sign_division[y] = Array2D_data[y]
            Array2D_data_sign_division[y, index_u] = -Array2D_data[y, index_u]
            Array2D_data_sign_division[y, index_v] = -Array2D_data[y, index_v]
        elif 270 < Array2D_data[y, index_d] <= 360.0:
            Array2D_data_sign_division[y] = Array2D_data[y]
            Array2D_data_sign_division[y, index_v] = -Array2D_data[y, index_v]
        else:
            print('Error: The value of the angle is out of range!')

    np.savetxt(fname=sign_division_output_path + '\\' + '2DDignDivision' + '_' + Num_tag + '.txt',
               X=Array2D_data_sign_division, fmt='%.32f')

    print('维度转换进度 {}/{}'.format(Progress_bar, file_nums))
    Progress_bar += 1


