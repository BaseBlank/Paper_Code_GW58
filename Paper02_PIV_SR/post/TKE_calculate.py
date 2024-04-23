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

    # Load all velocity data to compute the mean velocities
    velocities = []
    for file in tqdm(flow_matrix_files_sorted, desc='Loading data'):
        if file.endswith('.npy'):
            flow_matrix = np.load(os.path.join(args.input_path, file)).astype(np.float32)[:2]  # Only u and v
            velocities.append(flow_matrix)
    velocities = np.stack(velocities, axis=0)  # Shape [N, 2, H, W]

    # Mean velocities
    avg_velocities = np.mean(velocities, axis=0)  # Shape [2, H, W]

    # Calculate fluctuation and TKE
    u_prime_squared_sum = np.zeros_like(avg_velocities[0])
    v_prime_squared_sum = np.zeros_like(avg_velocities[1])

    # for velocity in tqdm(velocities, desc='Computing fluctuations'):
    #     u_prime_squared_sum += (velocity[0] - avg_velocities[0]) ** 2
    #     v_prime_squared_sum += (velocity[1] - avg_velocities[1]) ** 2

    for i, velocity in enumerate(tqdm(velocities, desc='Computing fluctuations and TKE')):
        u_prime = velocity[0] - avg_velocities[0]
        v_prime = velocity[1] - avg_velocities[1]

        # Saving pulsation velocities
        np.save(os.path.join(args.output_path_pulsation, f'u_prime_{i:04d}.npy'), u_prime)
        np.save(os.path.join(args.output_path_pulsation, f'v_prime_{i:04d}.npy'), v_prime)

        # Accumulate squared fluctuations
        u_prime_squared_sum += u_prime ** 2
        v_prime_squared_sum += v_prime ** 2

    u_prime_squared_avg = u_prime_squared_sum / total_steps
    v_prime_squared_avg = v_prime_squared_sum / total_steps

    tke = 0.5 * (u_prime_squared_avg + v_prime_squared_avg)

    # tke, shape: [H, W], 2D, to, [1, H, W], 3D, To convert to xytke txt data, plots in tecplot 360
    tke_3d = np.expand_dims(tke, axis=0)
    # save the TKE data to the output path
    np.savetxt(fname=args.output_path_tke + '/' + 'TKE_time_mean.txt', X=tke, fmt='%.32f')
    np.save(args.output_path_tke + '/' + 'TKE_time_mean.npy', tke_3d)

    # show the mean and max value of tke
    print(f"The mean value of TKE is: {np.mean(tke)}")
    print(f"The max value of TKE is: {np.max(tke)}")
    print(f"The shape of TKE is: {tke.shape}")


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




