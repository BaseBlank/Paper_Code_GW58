"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
# PSD analysis of velocity time series

# Code Reference Source
# https://scicoding.com/calculating-power-spectral-density-in-python/
# https://ww2.mathworks.cn/matlabcentral/answers/1860038-calculate-psd-using-fft2-for-2-d-matrix?s_tid=prof_contriblnk

import numpy as np
from scipy.signal import periodogram, welch
from scipy.fft import fft2, fftfreq, fftshift
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append('../')
from utils import sort_key


def PSD_calculating(args):
    """

    Args:
        args: Keyword parameters passed in
    """
    fs = 4000.0  # 4 kHz sampling frequency

    if args.one_dim_PSD == 'T':
        velocity_files = os.listdir(args.input_path_velocity)
        velocity_files_sorted = sorted(velocity_files, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                       reverse=False)

        # Two-dimensional distribution data of velocity field over a period of time
        # Initialize an empty list to store np arrays
        velocity_arrays = []
        for file in tqdm(velocity_files_sorted, desc='Preprocessing: ', colour="GREEN"):
            file_path = args.input_path_velocity + '/' + file
            Two_dim_velocity = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
            velocity_arrays.append(Two_dim_velocity)
        # Convert the list of arrays into a three-dimensional array
        Two_dim_velocity_period_time = np.array(velocity_arrays, dtype=np.float32)
        print('The two-dimensional velocity field data is spliced over a period of time')

        # One-dimensional PSD of the time series changes at each velocity point
        points_iteration = Two_dim_velocity_period_time.shape[1] * Two_dim_velocity_period_time.shape[2]
        # Use tqdm to create a progress bar
        progress_bar = tqdm(total=points_iteration, desc='One-dimensional Calculating PSD: ', colour='MAGENTA')

        p1 = 1
        for i in range(Two_dim_velocity_period_time.shape[1]):
            for j in range(Two_dim_velocity_period_time.shape[2]):
                velocity_series = Two_dim_velocity_period_time[:, Two_dim_velocity_period_time.shape[1]-i-1, j]
                # ('frequency [Hz]', 'PSD [V**2/Hz]'), The value range of f is actually [-0.5*fs, 0.5*fs]
                (f, Pxx_den) = periodogram(velocity_series, fs, scaling='density')
                PSD_result = np.vstack((f, Pxx_den)).T
                np.savetxt(
                    fname=args.output_1D_PSD + '/' + 'one-dim-PSD-i' + str(Two_dim_velocity_period_time.shape[1]-i-1) + 'j' + str(j) + '_' + sort_key(p1) + '.txt',
                    X=PSD_result, fmt='%.32f')
                p1 += 1

                # Update the progress bar after each iteration
                progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
    if args.one_dim_PSD == 'T':
        # free memory
        del Two_dim_velocity_period_time

    if args.two_dim_PSD == 'T':
        velocity_files = os.listdir(args.input_path_velocity)
        velocity_files_sorted = sorted(velocity_files, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                       reverse=False)

        p2 = 1
        for file in tqdm(velocity_files_sorted, desc='Two-dimensional Calculating PSD: ', colour="BLUE"):
            file_path = args.input_path_velocity + '/' + file
            Two_dim_velocity = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')

            fft_coefficients = fft2(Two_dim_velocity)  # Compute the 2D FFT

            # Compute the power spectrum by squaring the magnitude of the Fourier coefficients
            power_spectrum = np.abs(fft_coefficients) ** 2

            # Scale the power spectrum by the appropriate normalization factor
            # Normalize by the number of elements in the matrix
            two_dim_psd = power_spectrum / (Two_dim_velocity.shape[0] * fs * Two_dim_velocity.shape[1] * fs)

            np.savetxt(fname=args.output_2D_PSD + '/' + 'two-dim-PSD' + '_' + sort_key(p2) + '.txt',
                       X=two_dim_psd, fmt='%.32f')

            # Calculate the spatial frequencies
            freq_y = fftfreq(Two_dim_velocity.shape[0], 1/fs)
            freq_x = fftfreq(Two_dim_velocity.shape[1], 1/fs)

            p2 += 1
            if p2 == len(velocity_files_sorted) - 20:
                np.savetxt(fname=args.output_2D_PSD_axis + '/' + 'row-shape_0' + '.txt', X=freq_y, fmt='%.32f')
                np.savetxt(fname=args.output_2D_PSD_axis + '/' + 'column-shape_1' + '.txt', X=freq_x, fmt='%.32f')

    if args.one_dim_PSD_peak_detection == 'T':
        PSD_files_one_dim = os.listdir(args.output_1D_PSD)
        PSD_files_one_dim_sorted = sorted(PSD_files_one_dim, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                          reverse=False)

        p3 = 0
        one_dim_peak = []
        for file in tqdm(PSD_files_one_dim_sorted, desc='One-dimensional PSD peak detection: ', colour="YELLOW"):
            file_path = args.output_1D_PSD + '/' + file
            one_dim_result = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
            one_dim_frequency = one_dim_result[:, 0]
            one_dim_PSD = np.abs(one_dim_result[:, 1])
            max_idx = np.argmax(one_dim_PSD)
            largest_value = np.max(one_dim_PSD)  # Get the largest value
            sorted_values = np.sort(one_dim_PSD)  # Sort the array in ascending order
            second_last_largest_value = sorted_values[-2]  # Get the second-to-last largest value

            if largest_value >= 1.5 * second_last_largest_value:
                one_dim_peak.append(one_dim_result[max_idx, :])
            else:
                # one_dim_peak.append([one_dim_frequency[max_idx], 0])
                one_dim_peak.append(one_dim_result[max_idx, :])
                p3 += 1

        np.savetxt(fname=args.output_1D_PSD_peak_detection + '/' + 'one-dim-PSD-peak' + '_' + str(len(one_dim_peak)) + '.txt',
                   X=one_dim_peak, fmt='%.32f')
        print('Percentage of one-dim PSD peaks are not detected is: {}%'.format(100 * p3/len(PSD_files_one_dim_sorted)))

    if args.two_dim_PSD_peak_detection == 'T':
        PSD_files_two_dim = os.listdir(args.output_2D_PSD)
        PSD_files_two_dim_sorted = sorted(PSD_files_two_dim, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                          reverse=False)

        PSD_axis_y = np.loadtxt(os.path.join(args.output_2D_PSD_axis, 'row-shape_0.txt'),
                                dtype=np.float32, delimiter=' ')
        PSD_axis_x = np.loadtxt(os.path.join(args.output_2D_PSD_axis, 'column-shape_1.txt'),
                                dtype=np.float32, delimiter=' ')

        p4 = 0
        two_dim_peak = []
        for file in tqdm(PSD_files_two_dim_sorted, desc='Two-dimensional PSD peak detection: ', colour="RED"):
            file_path = args.output_2D_PSD + '/' + file
            two_dim_result = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')

            (row_index, column_index) = np.unravel_index(np.argmax(two_dim_result), two_dim_result.shape)

            largest_value = np.max(two_dim_result)  # Get the largest value
            sorted_values = np.sort(two_dim_result, axis=None)  # Sort the array in ascending order
            second_last_largest_value = sorted_values[-2]  # Get the second-to-last largest value

            if largest_value >= 1.5 * second_last_largest_value:
                two_dim_peak.append(np.array([PSD_axis_y[row_index], PSD_axis_x[column_index], largest_value]))
            else:
                # two_dim_peak.append(np.array([PSD_axis_y[row_index], PSD_axis_x[column_index], 0]))
                two_dim_peak.append(np.array([PSD_axis_y[row_index], PSD_axis_x[column_index], largest_value]))
                p4 += 1

        np.savetxt(fname=args.output_2D_PSD_peak_detection + '/' + 'two-dim-PSD-peak-value' + '.txt',
                   X=two_dim_peak, fmt='%.32f')
        print('Percentage of two-dim PSD peaks are not detected is: {}%'.format(100 * p4/len(PSD_files_two_dim_sorted)))

    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the pulsation of the 2D PIV flow field.")
    parser.add_argument("--one_dim_PSD", type=str,
                        help="Whether to process velocity information of one-dimensional time series.")
    parser.add_argument("--two_dim_PSD", type=str,
                        help="Whether to perform two-dimensional PSD on the entire flow field distribution data.")
    parser.add_argument("--one_dim_PSD_peak_detection", type=str,
                        help="Whether to determine the peak value of the calculated one-dimensional PSD.")
    parser.add_argument("--two_dim_PSD_peak_detection", type=str,
                        help="Whether to determine the peak value of the calculated two-dimensional PSD.")
    parser.add_argument("--input_path_velocity", type=str,
                        help="Velocity (in matrix form) folder location where PSD analysis is required.")
    parser.add_argument("--output_1D_PSD", type=str,
                        help="Save location of time series velocity 1D PSD results.")
    parser.add_argument("--output_2D_PSD", type=str,
                        help="Save the PSD result of the 2D flow field distribution.")
    parser.add_argument("--output_2D_PSD_axis", type=str,
                        help="Save the horizontal and vertical coordinates of the 2D PSD result.")
    parser.add_argument("--output_1D_PSD_peak_detection", type=str,
                        help="Save maximum peak information for 1D PSD.")
    parser.add_argument("--output_2D_PSD_peak_detection", type=str,
                        help="Save maximum peak information for 2D PSD.")

    args = parser.parse_args()

    PSD_calculating(args)

    # End of run, release all loaded cache and memory
    print('The program has been completed, and all resources have been released.')
    import gc

    gc.collect()
