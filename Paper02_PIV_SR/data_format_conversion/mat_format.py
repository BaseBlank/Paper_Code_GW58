"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
import numpy as np
import os
import scipy.io as scio
import argparse


def extract_data(args):
    """

    Args:
        args:
    """
    filepath = args.folder_path
    # filepath = 'D:/PIV/export/4000-0.324/mat/PIVlab.mat'
    # 加载.mat文件，返回字典数据
    dict_data = scio.loadmat(file_name=filepath, mdict=None, appendmat=True)

    # 提取对应的变量数据
    # 默认情况下，SciPy 将 MATLAB 结构读取为结构化 NumPy 数组，其中 dtype 字段的类型为object，
    array_data = dict_data[args.variable_name]

    Matrix_num = array_data.shape[0]
    print(Matrix_num)
    for i in range(Matrix_num):
        print(i)
        data_np_pre = array_data[i]
        data_np = data_np_pre[0]
        Num_tag = ' '
        if i < 10:
            Num_tag = '0000' + str(i)
        elif 10 <= i < 100:
            Num_tag = '000' + str(i)
        elif 100 <= i < 1000:
            Num_tag = '00' + str(i)
        elif 1000 <= i < 10000:
            Num_tag = '0' + str(i)
        else:
            Num_tag = str(i)

        np.savetxt('./data/PIV/' + args.variable_name + '_' + Num_tag + '.txt', data_np, fmt='%.32f', delimiter=',',
                   newline='\n', header='', footer='', comments='# ', encoding=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path", type=str, help="The folder location of the .mat file.")
    parser.add_argument("--variable_name", type=str, help="The name of the data variable to extract.")

    args = parser.parse_args()

    extract_data(args)

