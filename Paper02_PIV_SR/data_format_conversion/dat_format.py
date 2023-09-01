"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
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
    print('流场数据文件的数量为 {}'.format(File_num))

    for dat_file_name in dat_file_names_sorted:
        file_path = args.folder_path_input + '/' + dat_file_name
        data_np_void = np.genfromtxt(fname=file_path,
                                     skip_header=6,
                                     skip_footer=-1,
                                     names=["x", "y", "u", "v", "isNaN", "vorticity", "magnitude", "divergence", "dcev",
                                            "simple_shear", "simple_strain", "vector_direction"],
                                     dtype=np.float32,
                                     delimiter=' ')

        data_np = np.array(data_np_void.tolist(), dtype=np.float32)
        data_np_sel = np.take(data_np, indices=[2, 3, 11], axis=1)

        for i in range(data_np_sel.shape[0]):
            if data_np_sel[i, 2] < 0:
                data_np_sel[i, 2] += 360
        data_np_sel = np.abs(data_np_sel)

        # data, ["u", "v", "vector_direction"]
        data = np.zeros((59, 29, 3), dtype=np.float32)
        for c in range(3):
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    data[data.shape[0] - y - 1, x, c] = data_np_sel[y * data.shape[1] + x, c]

        if progression < 10:
            Num_tag = '0000' + str(progression)
        elif 10 <= progression < 100:
            Num_tag = '000' + str(progression)
        elif 100 <= progression < 1000:
            Num_tag = '00' + str(progression)
        elif 1000 <= progression < 10000:
            Num_tag = '0' + str(progression)
        else:
            Num_tag = str(progression)

        np.save(args.folder_path_output + '/' + args.variable_name + '_' + Num_tag + '.npy', data)

        print('转换进度 {}/{}'.format(progression, File_num))
        progression += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the .dat file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of save file.")
    parser.add_argument("--variable_name", type=str, help="The name of the data variable to extract.")

    args = parser.parse_args()

    extract_data(args)
