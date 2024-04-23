"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
import numpy as np
import os
import shutil
import argparse
import tqdm


# ==============================================================================
# 将xyzuv形式排列的excel多列表格形式的数据转换为矩阵形式（图像像素排列），用于SR模型的输入。
# ==============================================================================
def format_progression(progression):
    """
    格式化进度编号。
    """
    return f'0{progression:04d}' if progression < 10000 else str(progression)


def extract_data(args):
    """
    从指定文件夹提取并处理数据，然后保存到另一个文件夹中。
    Args:
        args: 一个包含输入文件夹路径（folder_path_input）和输出文件夹路径（folder_path_output）的对象，
              以及变量名（variable_name）。
    """
    if not os.path.exists(args.folder_path_output):
        # shutil.rmtree(args.folder_path_output)
        os.makedirs(args.folder_path_output)

    try:
        dat_file_names = os.listdir(args.folder_path_input)
    except FileNotFoundError:
        print(f"输入文件夹路径 {args.folder_path_input} 不存在。")
        return

    dat_file_names_sorted = sorted(dat_file_names, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                   reverse=False)

    file_nums = len(dat_file_names_sorted)
    print(f'流场数据文件的数量为 {file_nums}。')

    progress_bar = tqdm.tqdm(total=file_nums, desc='File conversion progress', unit='file')

    # for dat_file_name in dat_file_names_sorted:
    for progression, dat_file_name in enumerate(dat_file_names_sorted, start=1):
        file_path = os.path.join(args.folder_path_input, dat_file_name)
        try:
            data_np_void = np.genfromtxt(fname=file_path,
                                         skip_header=6,
                                         skip_footer=-1,
                                         names=["x", "y", "u", "v", "isNaN", "vorticity", "magnitude", "divergence",
                                                "dcev",
                                                "simple_shear", "simple_strain", "vector_direction"],
                                         dtype=np.float64,
                                         delimiter=' ')
        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")
            continue

        data_np = np.array(data_np_void.tolist(), dtype=np.float64)
        data_np_sel = np.take(data_np, indices=[2, 3, 11], axis=1)  # u, v, vector_direction

        # data_np_sel[:, 2] = np.where(data_np_sel[:, 2] < 0, data_np_sel[:, 2] + 360, data_np_sel[:, 2])
        # data_np_sel = np.abs(data_np_sel)

        # data, ["u", "v", "vector_direction"]
        DATA_DIM_0, DATA_DIM_1 = 44, 22
        data = np.zeros((3, DATA_DIM_0, DATA_DIM_1), dtype=np.float64)
        for c in range(data.shape[0]):
            for y in range(data.shape[1]):
                for x in range(data.shape[2]):
                    data[c, data.shape[1] - y - 1, x] = data_np_sel[y * data.shape[2] + x, c]

        num_tag = format_progression(progression)

        # output_file_path = os.path.join(args.folder_path_output, f"{args.variable_name}_{num_tag}.npy")
        output_file_path = os.path.join(args.folder_path_output, f"{args.variable_name}_{os.path.splitext(dat_file_name.split('_')[1])[0]}.npy")
        np.save(output_file_path, data, allow_pickle=False)

        progress_bar.set_description(f'File conversion progress {num_tag}/{file_nums}')
        progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Format Conversion scripts.")
    parser.add_argument("--folder_path_input", type=str, help="The folder location of the .dat file.")
    parser.add_argument("--folder_path_output", type=str, help="The folder location of save file.")
    parser.add_argument("--variable_name", type=str, help="The name of the data variable to extract.")

    args = parser.parse_args()

    extract_data(args)
