import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import sqrt
import imgproc

np.random.seed(928)
torch.manual_seed(928)
torch.cuda.manual_seed_all(928)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RecursiveBlock(nn.Module):
    def __init__(self, num_channels: int, num_residual_unit: int) -> None:
        super(RecursiveBlock, self).__init__()
        self.num_residual_unit = num_residual_unit

        self.residual_unit = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for _ in range(self.num_residual_unit):
            out = self.residual_unit(out)
            out = torch.add(out, x)

        return out


class DRRN(nn.Module):
    def __init__(self, num_residual_unit: int) -> None:
        super(DRRN, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1, 128, (3, 3), (1, 1), (1, 1), bias=False),
        )

        # Features trunk blocks
        self.trunk = RecursiveBlock(128, num_residual_unit)

        # Output layer
        self.conv2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 1, (3, 3), (1, 1), (1, 1), bias=False),
        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(identity, out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


sr_dir = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir'
gt_dir = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir'
sr_error_corr_dir = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/2DArray/SRDRNNerrorcorr_matrix_m'

fluids_files_sr = os.listdir(sr_dir)
fluids_files_sr_sorted = sorted(fluids_files_sr, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                reverse=False)
fluids_files_gt = os.listdir(gt_dir)
fluids_files_gt_sorted = sorted(fluids_files_gt, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                reverse=False)

progression = 1
File_num = int(len(fluids_files_sr_sorted))
print('The number of files for the correction calculation is {}'.format(File_num))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = DRRN().to(device)
model = DRRN(num_residual_unit=25).to(device)
model.load_state_dict(torch.load('./Error_correction_models/error_correction_DRRN_enetoend_model_finished.pth'))

for id in tqdm(range(File_num)):
    model.eval()
    file_path_sr = sr_dir + '/' + fluids_files_sr_sorted[id]
    fluids_sr_3D = np.load(file_path_sr).astype(np.float32)
    file_path_gt = gt_dir + '/' + fluids_files_gt_sorted[id]
    fluids_gt_3D = np.load(file_path_gt).astype(np.float32)

    sr_id = int(os.path.splitext(fluids_files_sr_sorted[id].split('_')[1])[0])
    gt_id = int(os.path.splitext(fluids_files_gt_sorted[id].split('_')[1])[0])
    assert sr_id == gt_id, 'The SR file id num and GT file id num is not equal.'

    Num_tag = os.path.splitext(fluids_files_sr_sorted[id].split('_')[1])[0]

    fluids_sr_3D[:, :, 2] = np.sqrt(fluids_sr_3D[:, :, 0] ** 2 + fluids_sr_3D[:, :, 1] ** 2)  # [H,W,3]
    fluids_gt_3D[:, :, 2] = np.sqrt(fluids_gt_3D[:, :, 0] ** 2 + fluids_gt_3D[:, :, 1] ** 2)  # [H,W,3]

    # max_value = np.max(fluids_gt_3D, axis=(0, 1))  # One-dimensional arrays
    # min_value = np.min(fluids_gt_3D, axis=(0, 1))
    # fluids_sr_3D_01 = imgproc.per_image_normalization(fluids_sr_3D)
    fluids_sr_3D_01 = fluids_sr_3D

    fluids_sr_3D_gpu = torch.from_numpy(fluids_sr_3D_01[:, :, 2:3]).to(device=device, dtype=torch.float32)  # [H,W,1]

    fluids_sr_3D_gpu_CHW = fluids_sr_3D_gpu.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]

    fluids_sr_3D_gpu_CHW_error_corr = model(fluids_sr_3D_gpu_CHW)  # [1,1,H,W]

    fluids_sr_3D_gpu_HWC_error_corr = fluids_sr_3D_gpu_CHW_error_corr.squeeze(0)  # [1,H,W]

    fluids_sr_3D_error_corr = fluids_sr_3D_gpu_HWC_error_corr.cpu().detach().numpy()

    # fluids_sr_3D_error_corr = fluids_sr_3D_error_corr * (max_value[2] - min_value[2]) + min_value[2]

    np.savetxt(sr_error_corr_dir + '/' + '2DMatrixSRDRNNcorrm_' + Num_tag + '.txt', fluids_sr_3D_error_corr[0, :, :],
               fmt='%.32f', delimiter=' ')
