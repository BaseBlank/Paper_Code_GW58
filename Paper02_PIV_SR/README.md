# RDN

> Ying T ,  Jian Y ,  Liu X . Image Super-Resolution via Deep Recursive Residual Network[C]// IEEE Conference on Computer Vision & Pattern Recognition. IEEE, 2017.

The network structure is visualized as follows: 



***Focus on this***

The code reference is from Lornatang's work contribution. Here again, to pay tribute to Lornatang's excellent work, the original reference code repository is as follows: 

> https://github.com/Lornatang/DRRN-PyTorch

Run data_format

```
python ./data_format_conversion/dat_format.py --folder_path_input F:/PIV_export/No_Heating/center_blockage_0.7/up/window_64_32_step_32/tecplot_file/include_derivatives --folder_path_output F:/PIV_export/No_Heating/center_blockage_0.7/up/window_64_32_step_32/model_data --variable_name PIV
```

Data division of training set and test set

```
python ./scripts/split_train_valid_dataset.py --train_images_dir F:/PIV_export/No_Heating/center_blockage_0.7/up/window_96_48_24_step_24/model_data --valid_images_dir F:/PIV_export/No_Heating/center_blockage_0.7/up/window_96_48_24_step_24/model_test --valid_samples_ratio 0.01
```

