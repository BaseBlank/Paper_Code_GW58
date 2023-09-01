**在进行模型train或者test的时候，一定要去config里调mode参数。**

**在进行模型train或者test的时候，一定要去config里调mode参数。**

**在进行模型train或者test的时候，一定要去config里调mode参数。**



将MATLAB的,mat格式转换为.txt格式

data_format_conversion.py

运行该程序，输入以下命令；

```shell
python ./data_format_conversion.py --folder_path D:/PIV/export/4000-0.324/mat/PIVlab.mat --variable_name u_filtered
```



infence推理计算：

```shell
python ./inference.py --model_arch_name rdn_small_x4 --inputs_path F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_data --output_path F:/model_generator/No_Heating/center_blockage_0.7/middle --upscale_factor 4 --model_weights_path F:/Code/RDN/results/RDN_small_x4-PIV/best.pth.tar --device_type cuda
```



test测试数据，baseline，插值



Test_dataset_acquisition.py

```shell
python data_format_conversion/Test_dataset_acquisition.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/model_test --folder_path_output_gt F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir --folder_path_output_lr F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir
```



推理结果维度转换

3D_2D_conversion.py

```shell
python ./3D_2D_conversion.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/middle --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/2DArray
```



Test_3D_2D_conversion

原始的GT数据

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT --variable_name 2DArrayGT
```

生成的SR数据，与原始的GT数据shape完全相同

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR --variable_name 2DArraySR
```

LR数据

```shell
python Test_3D_2D_conversion.py --folder_path_input_GT F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/LR --grid_scale 0.006537579617834 --Array2D_X_start 0.00628126326963875 --Array2D_Y_start 0.00759357749469175 --variable_name 2DArrayLR
```



sheet_to_matrix.py

GT

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/GT_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixGT
```

SR

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SR_matrix_m --extracted_variable m --matrix_rows_num_Y 56 --matrix_cols_num_X 28 --variable_name 2DMatrixSR
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

提取重建后的速度矢量转为矩阵（没改完）

```shell
python sheet_to_matrix.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/Re_matrix_V_Pulsation --extracted_variable v --matrix_rows_num_Y 236 --matrix_cols_num_X 116 --variable_name VPulsationMatrixRe
```

2D_matrix_interpolation.py

```shell
python 2D_matrix_interpolation.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/bicubic_matrix_m --interpolation_method bicubic
```



```shell
python 2D_matrix_interpolation.py --folder_path_input F:/model_generator/No_Heating/center_blockage_0.7/test_use/lr_dir --folder_path_output F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/linear_matrix_m --interpolation_method linear
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



脉动与时均速度计算

```shell
python Pulsation_volume.py --input_path_original_PIV F:/model_generator/No_Heating/center_blockage_0.7/2DArray_sign --output_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --output_path_mean F:/model_generator/No_Heating/center_blockage_0.7/2DArray_uv_mean
```

湍动能计算

重建数据

```shell
python TKE.py --input_path_Pulsation F:/model_generator/No_Heating/center_blockage_0.7/2DArray_Pulsation --output_path_k F:/model_generator/No_Heating/center_blockage_0.7/2DArray_TKE
```

原始PIV数据

```shell
python TKE.py --input_path_Pulsation F:/PIV_export/No_Heating/center_blockage_0.7/middle/window_96_48_24/tecplot/No_derivatives --output_path_k F:/model_generator/No_Heating/center_blockage_0.7/GT_2DArray_TKE --variable_name TKEGT --file_type dat
```

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

