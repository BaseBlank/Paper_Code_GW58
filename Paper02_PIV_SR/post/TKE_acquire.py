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
import shutil
from natsort import natsorted
import argparse
from tqdm import tqdm
import random


# Include these checks in the TKE computation loops for avg_u, avg_v, fluct_u, and fluct_v.
def check_inf_nan(arr, name="Array"):
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"{name} contains NaN or Inf values.")


def safe_square(x):
    """Safely square numbers while preventing overflow."""
    max_val = np.sqrt(np.finfo(np.float64).max)
    safe_x = np.clip(x, -max_val, max_val)
    return safe_x ** 2


def tke_computing(args):
    """
    Turbulent Kinetic Energy, TKE
    Args:
        args: Keyword parameters passed in
    """
    if os.path.exists(args.output_path_pulsation):
        shutil.rmtree(args.output_path_pulsation)
    os.makedirs(args.output_path_pulsation, mode=0o755)

    if os.path.exists(args.output_path_tke):
        shutil.rmtree(args.output_path_tke)
    os.makedirs(args.output_path_tke, mode=0o755)

    flow_matrix_files = os.listdir(args.input_path)  # .npy file
    flow_matrix_files_sorted = natsorted(flow_matrix_files)

    total_steps = int(len(flow_matrix_files_sorted))

    # Choose one at random from flow_matrix_files_sorted, use random
    file_random = args.input_path + '/' + random.choice(flow_matrix_files_sorted)
    try:
        flow_matrix_random = np.load(file_random).astype(np.float32)
    except ValueError:
        raise ValueError('The file type is not supported, please check the file type.')

    # Initialize matrices for storing average velocity components
    avg_u = np.empty((flow_matrix_random.shape[1], flow_matrix_random.shape[2]), dtype=np.float64)
    avg_v = np.empty((flow_matrix_random.shape[1], flow_matrix_random.shape[2]), dtype=np.float64)

    for file in tqdm(flow_matrix_files_sorted, desc='TKE Processing'):
        # check the file type, if it is .npy, use np.load to load the file
        assert file.endswith('.npy'), 'The file type is not supported, please check the file type.'
        # flow_matrix, shape: [3, H, W]; 3: [u, v, angles], u&v with positive and negative to indicate flow direction
        flow_matrix = np.load(args.input_path + '/' + file)[:2].astype(np.float32)

        # Update the running average of u and v
        avg_u += flow_matrix[0] / total_steps
        avg_v += flow_matrix[1] / total_steps

    # Subtract the average to get the fluctuating (or turbulent) components
    fluct_u = np.zeros_like(avg_u, dtype=np.float64)
    fluct_v = np.zeros_like(avg_v, dtype=np.float64)
    for file in tqdm(flow_matrix_files_sorted, desc='TKE Processing'):
        flow_data = np.load(args.input_path + '/' + file)[:2].astype(np.float32)  # shape: [2, H, W]
        check_inf_nan(flow_data[0], "Flow Data U")
        check_inf_nan(flow_data[1], "Flow Data V")

        # fluct_u += ((flow_data[0] - avg_u).astype(np.float64)) ** 2
        # fluct_v += ((flow_data[1] - avg_v).astype(np.float64)) ** 2
        fluct_u += safe_square(flow_data[0] - avg_u)
        fluct_v += safe_square(flow_data[1] - avg_v)

        # Save the pulsation rate by the way, Concatenate the two arrays pulsation_u and pulsation_v, [2, H, W]
        pulsation_u = flow_data[0] - avg_u
        pulsation_v = flow_data[1] - avg_v
        pulsation = np.concatenate((pulsation_u, pulsation_v), axis=0)  # shape: [2, H, W]
        np.save(args.output_path_pulsation + '/' + file, pulsation)

    check_inf_nan(fluct_u, "Fluctuation U")
    check_inf_nan(fluct_v, "Fluctuation V")
    # Calculate the turbulent kinetic energy (TKE) by taking the average over time
    tke = 0.5 * (fluct_u / total_steps + fluct_v / total_steps)  # shape: [H, W]

    # tke, shape: [H, W], 2D, to, [1, H, W], 3D, To convert to xytke txt data, plots in tecplot 360
    tke_3d = np.expand_dims(tke, axis=0)
    # save the TKE data to the output path
    np.savetxt(fname=args.output_path_tke + '/' + 'TKE_time_mean.txt', X=tke, fmt='%.32f')
    np.save(args.output_path_tke + '/' + 'TKE_time_mean.npy', tke_3d)

    # show the mean and max value of tke
    print(f"The mean value of TKE is: {np.mean(tke)}")
    print(f"The max value of TKE is: {np.max(tke)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Turbulent Kinetic Energy(TKE) of 2D PIV flow field.")
    parser.add_argument("--input_path", type=str,
                        help="The folder location of the PIV flow field pulsation velocity.")
    parser.add_argument("--output_path_pulsation", type=str,
                        help="The name of the pulsation variable to save txt file.")
    parser.add_argument("--output_path_tke", type=str,
                        help="The folder location of the processing out turbulent kinetic energy.")

    args = parser.parse_args()

    tke_computing(args)

    # End of run, release all loaded cache and memory
    print('The program has been completed, and all resources have been released.')
    import gc
    gc.collect()




