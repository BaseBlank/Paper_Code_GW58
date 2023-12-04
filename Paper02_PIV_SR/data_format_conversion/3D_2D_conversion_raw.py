"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================

# Raw PIV data computed directly from matlab is converted to standard 2D array form;
# Harmonized format to facilitate subsequent post processing.

import os
import numpy as np
from tqdm import tqdm


raw_folder_path = 'F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives'
output_folder_path = 'F:/model_generator/No_Heating/center_blockage_0.7/2DArray_raw_PIV'

dat_file_names = os.listdir(raw_folder_path)
dat_file_names_sorted = sorted(dat_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                               reverse=False)

File_num = int(len(dat_file_names_sorted))
print('The number of files that need to be extracted is {}'.format(File_num))

for file_name in tqdm(dat_file_names_sorted):
    file_path = raw_folder_path + '/' + file_name
    data_sheet = np.genfromtxt(fname=file_path,
                               skip_header=6,
                               skip_footer=-1,
                               names=["x", "y", "u", "v", "isNaN"],
                               dtype=np.float32,
                               delimiter=' ')
    data_sheet = np.array(data_sheet.tolist(), dtype=np.float32)

    data_fluid = np.zeros((data_sheet.shape[0], 6), dtype=np.float32)

    data_fluid[:, 0:4] = data_sheet[:, 0:4]
    data_fluid[:, 5] = np.sqrt(data_sheet[:, 2] ** 2 + data_sheet[:, 3] ** 2)

    Num_tag = os.path.splitext(file_name.split('_')[1])[0]

    np.savetxt(fname=output_folder_path + '/' + '2DArrayrawPIV' + '_' + Num_tag + '.txt', X=data_fluid, fmt='%.32f')

