"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
# Calculate the pulsation of the 2D PIV flow field

import numpy as np
import os
import argparse
from tqdm import tqdm
import time


def Pulsation_calculating(args):
    """

    Args:
        args: Keyword parameters passed in
    """
    global u_id, v_id
    original_files = os.listdir(args.input_path_original_PIV)
    original_files_sorted = sorted(original_files, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

    File_num = int(len(original_files_sorted))
    print('The number of flow field data files that need to be dimensionally transformed is {}'.format(File_num))

    index_x, index_y, index_u, index_v, index_d, index_M = 0, 1, 2, 3, 4, 5

    for file in tqdm(original_files_sorted, desc='mean Processing'):
        file_id = 0
        file_path = args.input_path_original_PIV + '/' + file
        original_PIV_2D = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')

        u_id = np.empty((original_PIV_2D.shape[0], File_num), dtype=np.float32)  # 只是元素无限小，并不是真空
        v_id = np.empty((original_PIV_2D.shape[0], File_num), dtype=np.float32)

        # u_id = np.concatenate((u_id, original_PIV_2D[:, index_u:index_v]), axis=1)
        # v_id = np.concatenate((v_id, original_PIV_2D[:, index_v:index_d]), axis=1)

        u_id[:, file_id:file_id+1] = original_PIV_2D[:, index_u:index_v]
        v_id[:, file_id:file_id+1] = original_PIV_2D[:, index_v:index_d]

        file_id += 1

        # time.sleep(0.1)

    u_mean = np.mean(u_id, axis=1)
    v_mean = np.mean(v_id, axis=1)

    u_mean = np.expand_dims(u_mean, axis=1)
    v_mean = np.expand_dims(v_mean, axis=1)

    with tqdm(total=File_num, desc='Calculation of progress', leave=True, unit='array', unit_scale=False) as pbar:
        for file in original_files_sorted:
            file_path = args.input_path_original_PIV + '/' + file
            original_PIV_2D = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')

            Num_tag = os.path.splitext(file.split('_')[1])[0]

            original_PIV_2D[:, index_u:index_v] = original_PIV_2D[:, index_u:index_v] - u_mean
            original_PIV_2D[:, index_v:index_d] = original_PIV_2D[:, index_v:index_d] - v_mean

            np.savetxt(fname=args.output_path_Pulsation + '/' + args.variable_name_Pulsation + '_' + Num_tag + '.txt',
                       X=original_PIV_2D, fmt='%.32f')

            pbar.update(1)  # 进度条更新
            # time.sleep()过小可能会引起不必要的开销，并有可能拖慢程序。最好是使用一个适合特定应用的较大的睡眠时间。
            # time.sleep(0.1)  # 可以完全删除，只是为了显示进度条

    file_path_random = args.input_path_original_PIV + '/' + original_files_sorted[0]
    original_PIV_2D = np.loadtxt(file_path_random, dtype=np.float32, delimiter=' ')
    uv_mean = np.concatenate((u_mean, v_mean), axis=1)
    original_PIV_2D[:, index_u:index_d] = uv_mean
    np.savetxt(fname=args.output_path_mean + '/' + args.variable_name_mean + '_' + str(File_num) + '.txt',
               X=original_PIV_2D, fmt='%.32f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the pulsation of the 2D PIV flow field.")
    parser.add_argument("--input_path_original_PIV", type=str,
                        help="The folder location of the original PIV flow field velocity.")
    parser.add_argument("--output_path_Pulsation", type=str,
                        help="The folder location of the processing out pulsation speed.")
    parser.add_argument("--output_path_mean", type=str,
                        help="The folder location of the hourly average speed.")
    parser.add_argument("--variable_name_Pulsation", type=str, default='UVPulsation',
                        help="The name of the pulsation variable to save txt file.")
    parser.add_argument("--variable_name_mean", type=str, default='UVmean',
                        help="The name of the mean variable to save txt file.")

    args = parser.parse_args()

    Pulsation_calculating(args)

    # End of run, release all loaded cache and memory
    print('The program has been completed, and all resources have been released.')
    import gc
    gc.collect()




