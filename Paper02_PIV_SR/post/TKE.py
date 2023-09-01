"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
# Calculate the Turbulence kinetic energy of the 2D PIV flow field

import numpy as np
import os
import argparse
from tqdm import tqdm


def TKE_calculating(args):
    """

    Args:
        args: Keyword parameters passed in
    """
    global TKE_id
    Pulsation_files = os.listdir(args.input_path_Pulsation)
    Pulsation_files_sorted = sorted(Pulsation_files, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                    reverse=False)

    totals = int(len(Pulsation_files_sorted))

    index_x, index_y, index_u, index_v, index_d, index_M = 0, 1, 2, 3, 4, 5

    for file in tqdm(Pulsation_files_sorted, desc='TKE Processing'):
        progression = 1
        file_path = args.input_path_Pulsation + '/' + file
        if args.file_type == 'txt':
            Pulsation_PIV_2D = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
        elif args.file_type == 'dat':
            Pulsation_PIV_2D = np.genfromtxt(fname=file_path,
                                             skip_header=6,
                                             skip_footer=-1,
                                             names=["x", "y", "u", "v", "isNaN"],
                                             dtype=np.float32,
                                             delimiter=' ')
            Pulsation_PIV_2D = np.array(Pulsation_PIV_2D.tolist(), dtype=np.float32)
        else:
            raise ValueError('The file type is not supported, please check the file type.')

        k = np.empty((Pulsation_PIV_2D.shape[0], 1), dtype=np.float32)  # 只是元素无限小，并不是真空
        for i in range(Pulsation_PIV_2D.shape[0]):
            # Here u and v are pulsating velocities, not directly measured instantaneous values.
            k[i] = 0.5 * (Pulsation_PIV_2D[i, index_u] ** 2 + Pulsation_PIV_2D[i, index_v] ** 2)

        TKE_id = np.empty((Pulsation_PIV_2D.shape[0], totals), dtype=np.float32)
        TKE_id[:, progression:progression+1] = k

    TKE = np.mean(TKE_id, axis=1)
    TKE = np.expand_dims(TKE, axis=1)

    file_random = args.input_path_Pulsation + '/' + Pulsation_files_sorted[0]
    if args.file_type == 'txt':
        Pulsation_PIV_2D = np.loadtxt(file_random, dtype=np.float32, delimiter=' ')
    elif args.file_type == 'dat':
        Pulsation_PIV_2D = np.genfromtxt(fname=file_random,
                                         skip_header=6,
                                         skip_footer=-1,
                                         names=["x", "y", "u", "v", "isNaN"],
                                         dtype=np.float32,
                                         delimiter=' ')
        Pulsation_PIV_2D = np.array(Pulsation_PIV_2D.tolist(), dtype=np.float32)
    else:
        raise ValueError('The file type is not supported, please check the file type.')

    TKE_write = np.concatenate((Pulsation_PIV_2D[:, index_x:index_u], TKE), axis=1)

    np.savetxt(fname=args.output_path_k + '/' + args.variable_name + '_' + str(totals) + '.txt',
               X=TKE_write, fmt='%.32f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the pulsation of the 2D PIV flow field.")
    parser.add_argument("--input_path_Pulsation", type=str,
                        help="The folder location of the PIV flow field pulsation velocity.")
    parser.add_argument("--output_path_k", type=str,
                        help="The folder location of the processing out turbulent kinetic energy.")
    parser.add_argument("--variable_name", type=str, default='TKE',
                        help="The name of the pulsation variable to save txt file.")
    parser.add_argument("--file_type", type=str, default='txt',
                        help="The input data file format to be processed.")

    args = parser.parse_args()

    TKE_calculating(args)

    # End of run, release all loaded cache and memory
    print('The program has been completed, and all resources have been released.')
    import gc
    gc.collect()




