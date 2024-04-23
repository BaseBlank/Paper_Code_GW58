



**在进行模型train或者test的时候，一定要去config里调mode参数。**

**在进行模型train或者test的时候，一定要去config里调mode参数。**

**在进行模型train或者test的时候，一定要去config里调mode参数。**



# 一、处理原始软件计算的数据

这只是个参考，用的是middle中心区域的数据，没有用下面示例中的up区域。

1.1 从软件计算出来最原始的数据处理为模型需要的数据，这里使用的teceplot文件---->.npy文件。

```shell
python ./data_format_conversion/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/up/window_64_32_step_32/tecplot_file/include_derivatives --folder_path_output F:/PIV_export/No_Heating/center_blockage_0.7/up/window_64_32_step_32/model_data --variable_name PIV
```

1.2 划分训练集与测试集

```shell
python ./scripts/split_train_valid_dataset.py --train_images_dir F:/PIV_export/No_Heating/center_blockage_0.7/up/window_96_48_24_step_24/model_data --valid_images_dir F:/PIV_export/No_Heating/center_blockage_0.7/up/window_96_48_24_step_24/model_test --valid_samples_ratio 0.01
```

# 二、模型训练

相关设置直接在config.py文件中完成。







导出了多种格式，包括.mat以及teceplot格式等，用哪一种格式提取数据都行。

将MATLAB的,mat格式转换为.txt格式

data_format_conversion.py

运行该程序，输入以下命令；

```shell
python ./data_format_conversion.py --folder_path D:/PIV/export/4000-0.324/mat/PIVlab.mat --variable_name u_filtered
```

上面的应该错了；

```shell
python ./mat_format.py --folder_path D:/PIV/export/4000-0.324/mat/PIVlab.mat --variable_name u_filtered
```







# 三、infence推理计算

3.1 模型训练完了，将模型直接用在原始尺寸的软件处理转换后的.npy数据上。

```shell
python ./inference.py --model_arch_name rdn_small_x4 --inputs_path F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_data --output_path F:/model_generator/No_Heating/center_blockage_0.7/middle --upscale_factor 4 --model_weights_path F:/Code/RDN/results/RDN_small_x4-PIV/best.pth.tar --device_type cuda
```

~~用在测试数据上~~

```shell
python ./inference_opt.py --model_arch_name rdn_small_x4 --inputs_path F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --maxmin_path F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir --output_path F:/model_generator/No_Heating/center_blockage_0.7/middle --upscale_factor 4 --model_weights_path F:/Code/RDN/results/RDN_small_x4-PIV/best.pth.tar --device_type cuda
```



推理结果维度转换

3D_2D_conversion.py

```shell
python ./3D_2D_conversion.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/middle --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/2DArray
```



# 四、基于test数据做SR误差计算

test测试数据，baseline，插值



**4.1 将上面划分的测试集，现在是原始尺寸[59,29,3]，处理为用于SR计算的[56,28,3]的尺寸，以及采用下采样算法image_resize下采样得到的LR数据，目前获得都是.npy数据。**

Test_dataset_acquisition.py

```shell
python data_format_conversion/Test_dataset_acquisition.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_test --folder_path_output_gt F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir --folder_path_output_lr F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir
```

**4.2 通过传统数学算法将LR数据上采样到SR数据**

2D_matrix_interpolation.py

```shell
python 2D_matrix_interpolation.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/bicubic_matrix_m --interpolation_method bicubic
```



```shell
python 2D_matrix_interpolation.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/linear_matrix_m --interpolation_method linear
```



**4.3 将用于SR误差定量测试的.npy数据，转为二维数组，应该是为了画图的前期准备。**

Test_3D_2D_conversion

原始的GT数据

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT --variable_name 2DArrayGT
```

生成的SR数据，与原始的GT数据shape完全相同

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR --variable_name 2DArraySR
```

校准后的SR数据

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_opt_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRerrorcorr --variable_name 2DArraySRcorr
```

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_opt_end_to_end_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRDRNNerrorcorr --variable_name 2DArraySRDRRNcorr
```

LR数据

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/LR --grid_scale 0.006537579617834 --Array2D_X_start 0.00628126326963875 --Array2D_Y_start 0.00759357749469175 --variable_name 2DArrayLR
```

**4.4 为了在origin画图，将["x", "y", "u", "v", "vector_direction", "vector_magnitude"]的txt数据去掉坐标转为矩阵形式的txt，完成origin的云图绘制。**

sheet_to_matrix.py

GT

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixGT
```

SR

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixSR
```

修正过的SR

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRerrorcorr --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRerrorcorr_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixSRcorr
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRDRNNerrorcorr --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRDENNerrorcorr_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixSRDRNNcorr
```

LR

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/LR --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/LR_matrix_m --extracted_variable m --matrix_rows_num_Y 14 --matrix_cols_num_X 7 --variable_name 2DMatrixLR
```



提取重建后的u与v速度转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_u --extracted_variable u --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name 2DMatrixRe
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_v --extracted_variable v --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name 2DMatrixRe
```

提取原始PIV的u与v速度转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/GT_matrix_u --extracted_variable u --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name 2DMatrixGT --file_type dat
```

```shell
python sheet_to_matrix.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/GT_matrix_v --extracted_variable v --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name 2DMatrixGT --file_type dat
```



提取重建后的u与v时均速度转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_u_mean --extracted_variable u --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name UMeanMatrixRe
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_v_mean --extracted_variable v --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name VMeanMatrixRe
```

提取重建后的u与v脉动速度转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_U_Pulsation --extracted_variable u --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name UPulsationMatrixRe
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_V_Pulsation --extracted_variable v --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name VPulsationMatrixRe
```



提取**重建前的原始PIV的u与v时均速度**转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean_raw_PIV --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Raw_matrix_u_mean --extracted_variable u --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name uMeanMatrixRaw
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean_raw_PIV --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Raw_matrix_v_mean --extracted_variable v --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name VMeanMatrixRaw
```

提取**重建前的原始PIV的u与v脉动速度**转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_raw_PIV --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Raw_matrix_U_Pulsation --extracted_variable u --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name UPulsationMatrixRaw
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_raw_PIV --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Raw_matrix_V_Pulsation --extracted_variable v --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name VPulsationMatrixRaw
```

提取重建后的速度矢量转为矩阵（没改完）

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_V_Pulsation --extracted_variable v --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name VPulsationMatrixRe
```



vortexfitting

```shell
python run.py --input C:\\Users\\1\\Desktop\\exampla\\2DArrayGT_00062.txt --output C:\\Users\\1\\Desktop\\exampla\\output --scheme 2 --detect Q --threshold 0.0 --boxsize 6 --flip 0 --meanfilename / --plot fit --filetype Reconstructed_2D_data
```



Vortex_Identification_Criteria

重建数据

```shell
python Vortex_Identification_Criteria.py --re_results_folder F:\\model_generator\\No_Heating\\center_blockage_0.7\\2DArray_sign --vortex_identification_folder F:\\model_generator\\No_Heating\\center_blockage_0.7\\Vortex\Omega\\Matrix_form --finite_difference_scheme 2 --detection_method Omega
```

```shell
python Vortex_Identification_Criteria.py --re_results_folder F:\\model_generator\\No_Heating\\center_blockage_0.7\\2DArray_sign --vortex_identification_folder F:\\model_generator\\No_Heating\\center_blockage_0.7\\Vortex\\Q\\Matrix_form --finite_difference_scheme 2 --detection_method Q
```

原始PIV数据

```shell
python Vortex_Identification_Criteria.py --re_results_folder F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives --vortex_identification_folder F:\\model_generator\\No_Heating\\center_blockage_0.7\\Vortex\Omega_GT\\Matrix_form --finite_difference_scheme 2 --detection_method Omega  --file_type dat
```



**脉动与时均速度计算**

```shell
python Pulsation_volume.py --input_path_original_PIV F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --output_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --output_path_mean F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean
```

**原始PIV数据的脉动与时均速度计算**

首先从3D数据获取2D数据

原始的3D数据在这里

> ```
> F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_data
> ```

但是3D_2D_conversion.py默认坐标是按照超分后的236的尺寸写的，还得从头写，直接再一个脚本3D_2D_conversion_raw.py吧。



2D数据划分象限

并且由于原始MATLAB计算出的原始PIV速度自带正负号，所以象限转换也省了，在3D_2D_conversion_raw.py中一起包含了。

进行脉动计算

```shell
python Pulsation_volume.py --input_path_original_PIV F:/model_generator/No_Heating/center_blockage_0.7/2DArray_raw_PIV --output_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation_raw_PIV --output_path_mean F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean_raw_PIV
```



**湍动能计算**

重建数据

```shell
python TKE.py --input_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --output_path_k F:/model_generator/No_Heating/center_blockage_0.7/2DArray_TKE
```

原始PIV数据

```shell
python TKE.py --input_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation_raw_PIV --output_path_k F:/model_generator/No_Heating/center_blockage_0.7/GT_2DArray_TKE --variable_name TKEGT
```



python TKE.py --input_path_Pulsation ~~F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives~~ --output_path_k F:/model_generator/No_Heating/center_blockage_0.7/GT_2DArray_TKE --variable_name TKEGT --file_type dat

湍动能结果2D转矩阵

重建数据

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_TKE --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_TKE --extracted_variable u --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name TKEMatrixRe
```

原始PIV数据

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/GT_2DArray_TKE --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/GT_matrix_TKE --extracted_variable u --matrix_rows_num_Y 59 --matrix_cols_num_X 29 --variable_name TKEMatrixGT
```



PSD分析

重建速度场

标量速度u计算

```shell
python flow_PSD.py --one_dim_PSD T --two_dim_PSD T --one_dim_PSD_peak_detection T --two_dim_PSD_peak_detection T --input_path_velocity F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_u --output_1D_PSD F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/one_dim --output_2D_PSD F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/two_dim --output_2D_PSD_axis F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/two_dim_axis --output_1D_PSD_peak_detection F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/one_dim_peak --output_2D_PSD_peak_detection F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/two_dim_peak
```

标量速度v计算

```shell
python flow_PSD.py --one_dim_PSD T --two_dim_PSD T --one_dim_PSD_peak_detection T --two_dim_PSD_peak_detection T --input_path_velocity F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_v --output_1D_PSD F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/one_dim --output_2D_PSD F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/two_dim --output_2D_PSD_axis F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/two_dim_axis --output_1D_PSD_peak_detection F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/one_dim_peak --output_2D_PSD_peak_detection F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/two_dim_peak
```

将PSD结果转为矩阵

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/one_dim_peak --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/U/one_dim_peak_matrix --extracted_variable x --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name UPSDpeakMatrix
```

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/one_dim_peak --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/PSD/Re/V/one_dim_peak_matrix --extracted_variable x --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name VPSDpeakMatrix
```

# 标准计算流程：

# 一、数据预处理

## 1.1 数据格式转换

PIV计算导出tecplot文件，首先将二维的.dat文件转换为三维的类似图片像素排列的.npy数据；

形成dataset文件夹

`dat_format.py`

```bash
python ./scripts/dat_format.py --folder_path_input D:\\Code\\PIV_data\\flow1 --folder_path_output D:\\Code\\PIV_dataset\\dataset --variable_name PIV-1
```

```bash
python ./scripts/dat_format.py --folder_path_input D:\\Code\\PIV_data\\flow2 --folder_path_output D:\\Code\\PIV_dataset\\dataset --variable_name PIV-2
```

`dat_format.py`中需要验证的部分

|         代码中易错需要验证的部分         | 是否验证 |
| :--------------------------------------: | :------: |
|    三列数据到三通道矩阵像素数据的转换    |  已验证  |
| 读取原始数据文件夹后文件进行排序是否正确 |  已验证  |

## 1.2 数据集划分

划分训练集、验证集与测试集

形成train_dataset、validate_dataset、test_dataset三个文件夹。

使用全部数据

```bash
python ./scripts/split_train_valid_dataset.py --raw_dataset_dir D:/Code/PIV_dataset/dataset --train_dataset_dir D:/Code/PIV_dataset/train_dataset --valid_dataset_dir D:/Code/PIV_dataset/validate_dataset --test_dataset_dir D:/Code/PIV_dataset/test_dataset --valid_samples_ratio 0.2 --test_samples_ratio 0.1 --all_flow_conditions_used
```

不使用全部数据

```bash
python ./scripts/split_train_valid_dataset.py --raw_dataset_dir D:/Code/PIV_dataset/dataset --train_dataset_dir D:/Code/PIV_dataset/train_dataset --valid_dataset_dir D:/Code/PIV_dataset/validate_dataset --test_dataset_dir D:/Code/PIV_dataset/test_dataset --valid_samples_ratio 0.2 --test_samples_ratio 0.1
```

## 1.3 数据shpae裁剪

数据大小预处理

形成train、validate、test三个文件夹。

```bash
python ./scripts/prepare_dataset.py --dataset_dir D:/Code/PIV_dataset/train_dataset --output_dir D:/Code/PIV_dataset/train --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir D:/Code/PIV_dataset/validate_dataset --output_dir D:/Code/PIV_dataset/validate --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir D:/Code/PIV_dataset/test_dataset --output_dir D:/Code/PIV_dataset/test --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```



## 1.4 归一化最值计算

计算最大最小值，运行scripts/minimax_search.py。

## 1.5 生成用于验证与测试的gt-lr数据对

生成validate_lr、validate_gt、test_lr、test_gt四个文件夹；

不执行归一化的话，validate_gt、test_gt两个文件夹与validate、test两个文件夹相同，执行归一化，那么validate_gt、test_gt两个文件夹就是validate、test两个文件夹的归一化版本。

因为train的数据在Dataloader中直接成对的进行了resize，因此不需要要实现准备LR数据。

```bash
python ./scripts/GT_LR_generate.py --gt_dir D:/Code/PIV_dataset/validate --lr_dir_normalize D:/Code/PIV_dataset/validate_lr --gt_dir_normalize D:/Code/PIV_dataset/validate_lr_gt --down_sampling_mode random --normalized_choice normalized_used --down_sampling_factor 4 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/GT_LR_generate.py --gt_dir D:/Code/PIV_dataset/test --lr_dir_normalize D:/Code/PIV_dataset/test_lr --gt_dir_normalize D:/Code/PIV_dataset/test_lr_gt --down_sampling_mode random --normalized_choice normalized_unused --down_sampling_factor 4 --all_flow_conditions_used --num_workers 4
```

## 二、计算输入通道修改

需要修改的内容包括：

dataset.py文件的两个数据加载函数；

config.py文件的输入输出通道数配置；

## 三、推理计算

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_20240328/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/test_lr --output_path F:/PIV_model_generate/PIV_dataset/test_lr_sr --upscale_factor 4 
```

推理计算数据

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_20240328/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/test_lr --output_path F:/PIV_model_generate/PIV_dataset/test_lr_sr --upscale_factor 4 
```

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path D:/Code/RDNopt/results/rdn_large_x4-PIV/best.pth.tar --inputs_path D:/Code/PIV_dataset/test_lr --output_path D:/Code/PIV_dataset/test_lr_sr --upscale_factor 4 
```



## 计算后处理

### **数学算法将LR数据上采样到SR数据**

2D_matrix_interpolation.py

```shell
python post/test_interpolation.py --folder_path_input D:/Code/PIV_dataset/test_lr --folder_path_output D:/Code/PIV_dataset/test_lr_interpolation --interpolation_method random
```



三维数组到二维数组的转换

```bash
python post/3D_2D_conversion.py --folder_path_input D:/Code/PIV_dataset/dataset --folder_path_output D:/Code/PIV_data/flow_2d
```



```bash
python post/TKE_acquire.py --input_path D:/Code/PIV_dataset/dataset --output_path_pulsation D:/Code/PIV_dataset/dataset_pulsation --output_path_tke D:/Code/PIV_dataset/dataset_TKE
```



# 实际计算参数设置流程：

# 一、数据预处理

## 1.1 数据格式转换

PIV计算导出tecplot文件，首先将二维的.dat文件转换为三维的类似图片像素排列的.npy数据；

形成dataset文件夹

`dat_format.py`

`window_64_32-4000_0.324_downstream`文件夹，0.324的流动工况；

```bash
python ./scripts/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_64_32-4000_0.324_downstream/tecplot --folder_path_output F:/PIV_model_generate/PIV_dataset/dataset --variable_name PIV-1
```

`window_64_32-4000_0.27_downstream`文件夹，0.27的流动工况；

```bash
python ./scripts/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_64_32-4000_0.27_downstream/tecplot --folder_path_output F:/PIV_model_generate/PIV_dataset/dataset --variable_name PIV-2
```

`window_64_32-4000_0.162_downstream`文件夹，0.162的流动工况；

```bash
python ./scripts/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_64_32-4000_0.162_downstream/tecplot --folder_path_output F:/PIV_model_generate/PIV_dataset/dataset --variable_name PIV-3
```

`window_64_32-4000_0.216_downstream`文件夹，0.216的流动工况；

```bash
python ./scripts/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_64_32-4000_0.216_downstream/tecplot --folder_path_output F:/PIV_model_generate/PIV_dataset/dataset --variable_name PIV-4
```

`window_64_32-4000_0.108_downstream`文件夹，0.108的流动工况；

```bash
python ./scripts/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_64_32-4000_0.108_downstream/tecplot --folder_path_output F:/PIV_model_generate/PIV_dataset/dataset --variable_name PIV-5
```

`dat_format.py`中需要验证的部分

|         代码中易错需要验证的部分         | 是否验证 |
| :--------------------------------------: | :------: |
|    三列数据到三通道矩阵像素数据的转换    |  已验证  |
| 读取原始数据文件夹后文件进行排序是否正确 |  已验证  |

## 1.2 数据集划分

划分训练集、验证集与测试集

形成train_dataset、validate_dataset、test_dataset三个文件夹。

使用全部数据

```bash
python ./scripts/split_train_valid_dataset.py --raw_dataset_dir D:/Code/PIV_dataset/dataset --train_dataset_dir D:/Code/PIV_dataset/train_dataset --valid_dataset_dir D:/Code/PIV_dataset/validate_dataset --test_dataset_dir D:/Code/PIV_dataset/test_dataset --valid_samples_ratio 0.2 --test_samples_ratio 0.1 --all_flow_conditions_used
```

不使用全部数据

```bash
python ./scripts/split_train_valid_dataset.py --raw_dataset_dir F:/PIV_model_generate/PIV_dataset/dataset --train_dataset_dir F:/PIV_model_generate/PIV_dataset/train_dataset --valid_dataset_dir F:/PIV_model_generate/PIV_dataset/validate_dataset --test_dataset_dir F:/PIV_model_generate/PIV_dataset/test_dataset --valid_samples_ratio 0.2 --test_samples_ratio 0.1
```

使用全部数据

```bash
python ./scripts/split_train_valid_dataset.py --raw_dataset_dir F:/PIV_model_generate/PIV_dataset/dataset --train_dataset_dir F:/PIV_model_generate/PIV_dataset/train_dataset --valid_dataset_dir F:/PIV_model_generate/PIV_dataset/validate_dataset --test_dataset_dir F:/PIV_model_generate/PIV_dataset/test_dataset --valid_samples_ratio 0.2 --test_samples_ratio 0.1 --all_flow_conditions_used
```



## 1.3 数据shpae裁剪

数据大小预处理

形成train、validate、test三个文件夹。

使用全部数据

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/train_dataset --output_dir F:/PIV_model_generate/PIV_dataset/train --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/validate_dataset --output_dir F:/PIV_model_generate/PIV_dataset/validate --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/test_dataset --output_dir F:/PIV_model_generate/PIV_dataset/test --flow_crop_height 44 --flow_crop_width 20 --all_flow_conditions_used --num_workers 4
```

不使用全部数据

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/train_dataset --output_dir F:/PIV_model_generate/PIV_dataset/train --flow_crop_height 44 --flow_crop_width 20 --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/validate_dataset --output_dir F:/PIV_model_generate/PIV_dataset/validate --flow_crop_height 44 --flow_crop_width 20 --num_workers 4
```

```bash
python ./scripts/prepare_dataset.py --dataset_dir F:/PIV_model_generate/PIV_dataset/test_dataset --output_dir F:/PIV_model_generate/PIV_dataset/test --flow_crop_height 44 --flow_crop_width 20 --num_workers 4
```



## 1.4 归一化最值计算

计算最大最小值，运行scripts/minimax_search.py。

## 1.5 生成用于验证与测试的gt-lr数据对

生成validate_lr、validate_gt、test_lr、test_gt四个文件夹；

不执行归一化的话，validate_gt、test_gt两个文件夹与validate、test两个文件夹相同，执行归一化，那么validate_gt、test_gt两个文件夹就是validate、test两个文件夹的归一化版本。

因为train的数据在Dataloader中直接成对的进行了resize，因此不需要要实现准备LR数据。

不使用全部数据

验证集，全在01的范围

```bash
python ./scripts/GT_LR_generate.py --gt_dir F:/PIV_model_generate/PIV_dataset/validate --lr_dir_normalize F:/PIV_model_generate/PIV_dataset/validate_lr --gt_dir_normalize F:/PIV_model_generate/PIV_dataset/validate_lr_gt --down_sampling_mode random --normalized_choice normalized_used --down_sampling_factor 4 --num_workers 4
```

测试集，LR在01范围，GT不在，保证先归一化再下采样，而不是先下采样再归一化。

```bash
python ./scripts/GT_LR_generate.py --gt_dir F:/PIV_model_generate/PIV_dataset/test --lr_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr --gt_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr_gt --down_sampling_mode random --normalized_choice normalized_unused --down_sampling_factor 4 --num_workers 4
```

```bash
python ./scripts/GT_LR_generate.py --gt_dir F:/PIV_model_generate/PIV_dataset/test --lr_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr --gt_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr_gt --down_sampling_mode pooling --normalized_choice normalized_used --down_sampling_factor 4 --num_workers 4
```

使用了全部数据

```bash
python ./scripts/GT_LR_generate.py --gt_dir F:/PIV_model_generate/PIV_dataset/validate --lr_dir_normalize F:/PIV_model_generate/PIV_dataset/validate_lr --gt_dir_normalize F:/PIV_model_generate/PIV_dataset/validate_lr_gt --down_sampling_mode random --normalized_choice normalized_used --down_sampling_factor 4 --all_flow_conditions_used --num_workers 4
```

```bash
python ./scripts/GT_LR_generate.py --gt_dir F:/PIV_model_generate/PIV_dataset/test --lr_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr --gt_dir_normalize F:/PIV_model_generate/PIV_dataset/test_lr_gt --down_sampling_mode random --normalized_choice normalized_unused --down_sampling_factor 4 --all_flow_conditions_used --num_workers 4
```

# 二、计算输入通道修改

需要修改的内容包括：

dataset.py文件的两个数据加载函数；

config.py文件的输入输出通道数配置；

# 三、推理计算推理计算数据

对比几种变量对于最终测试结果的影响，训练集、验证集和测试集严格按照7：2：1的比例划分，都是速度模的误差，暂时没记录速度分量：

|     模型     | 输出01限制 | 平均误差 | L2误差 |
| :----------: | :--------: | :------: | :----: |
| best.pth.tar |     否     |  0.2911  | 0.1997 |
| last.pth.tar |     否     |  0.3310  | 0.2118 |
| best.pth.tar |     是     |  0.2941  | 0.1984 |
| last.pth.tar |     是     |  0.3334  | 0.2125 |

**推理计算测试test数据**

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_20240408/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/test_lr --output_path F:/PIV_model_generate/PIV_dataset/test_lr_sr --upscale_factor 4
```

**推理计算所有源数据**

```bash
python inference_opt.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_20240415_all_dataset_pooling_1000/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/dataset --output_path F:/PIV_model_generate/PIV_dataset/dataset_sr --upscale_factor 4
```

```bash
python inference_opt.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code\RDNopt/experimental_records/Single_flow_condition_20240415_all_dataset_cubic_1000/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/dataset --output_path F:/PIV_model_generate/PIV_dataset/dataset_sr --upscale_factor 4
```

**推理计算测试test数据**，带输出缩放的模型

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_01_20240328/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/test_lr --output_path F:/PIV_model_generate/PIV_dataset/test_lr_sr --upscale_factor 4
```

**推理计算所有源数据**，带输出缩放的模型

```bash
python inference.py --device_type cuda --model_arch_name rdn_large_x4 --model_weights_path F:/Code/RDNopt/experimental_records/Single_flow_condition_20240408/results/rdn_large_x4-PIV/best.pth.tar --inputs_path F:/PIV_model_generate/PIV_dataset/dataset --output_path F:/PIV_model_generate/PIV_dataset/dataset_sr --upscale_factor 4
```

**数学算法将LR数据上采样到SR数据**

**数学算法将LR数据上采样到SR数据**

2D_matrix_interpolation.py

```shell
python post/test_interpolation.py --folder_path_input F:/PIV_model_generate/PIV_dataset/test_lr --folder_path_output F:/PIV_model_generate/PIV_dataset/test_lr_interpolation --interpolation_method random
```

# 四、计算衍生参数

计算TKE值

原始数据

```bash
python post/TKE_calculate.py --input_path F:/PIV_model_generate/PIV_dataset/dataset --output_path_pulsation F:/PIV_model_generate/PIV_dataset/dataset_pulsation --output_path_tke F:/PIV_model_generate/PIV_dataset/dataset_TKE
```

重建数据

```bash
python post/TKE_calculate.py --input_path F:/PIV_model_generate/PIV_dataset/dataset_sr --output_path_pulsation F:/PIV_model_generate/PIV_dataset/dataset_sr_pulsation --output_path_tke F:/PIV_model_generate/PIV_dataset/dataset_sr_TKE
```

计算出来的TKE矩阵转为XYZQ结构

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/dataset_TKE --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/dataset_TKE --if_velocity_calculation V_Flase --if_normalized_velocity V_
```



```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/dataset_sr_TKE --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/dataset_sr_TKE --if_velocity_calculation V_Flase --if_normalized_velocity V_
```

test数据集中的LR矩阵转为XYZQ结构

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/test_lr --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/test_lr --if_velocity_calculation V_True --if_normalized_velocity V_nor
```

test数据集中的GT矩阵转为XYZQ结构，归一化的数据

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/test_lr_gt --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/test_lr_gt --if_velocity_calculation V_True --if_normalized_velocity V_nor
```

test数据集中的SR矩阵转为XYZQ结构，已经乘过了，不乘了

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/test_lr_sr --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/test_lr_sr --if_velocity_calculation V_True --if_normalized_velocity _
```

test数据集中的interpolation矩阵转为XYZQ结构，已经乘过了，不乘了

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/test_lr_interpolation --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/test_lr_interpolation --if_velocity_calculation V_True --if_normalized_velocity _
```

所有数据集中的SR矩阵转为XYZQ结构，已经乘过了，不乘了

```bash
python post/3D_2D_conversion.py --folder_path_input F:/PIV_model_generate/PIV_dataset/dataset_sr --folder_path_output F:/PIV_model_generate/PIV_dataset/XYZQ/dataset_sr --if_velocity_calculation V_True --if_normalized_velocity _
```

直接保存
