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


class error_calibration_module(nn.Module):
    """
    Very simple error correction module, does it work better?
    The structure is basically similar to that of the SRCNN model
    """

    def __init__(self, input_channels, num_channels, deeper=False):
        super(error_calibration_module, self).__init__()
        self.deeper = deeper
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        if self.deeper:
            x1 = self.conv1(X)
            x2 = F.relu(x1)
            x3 = self.conv3(x2)
            x4 = F.relu(x3)
            x5 = self.conv2(x4)

            return x5
        else:
            x1 = self.conv1(X)
            x2 = F.relu(x1)
            x3 = self.conv2(x2)

            return x3


class error_calibration_module_large_kernel(nn.Module):
    def __init__(self, input_channels, num_channels):
        super(error_calibration_module_large_kernel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv2d(num_channels, num_channels//2, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(num_channels//2, input_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = F.relu(x1, inplace=True)
        x3 = self.conv3(x2)
        x4 = F.relu(x3, inplace=True)
        x5 = self.conv2(x4)

        return x5


class DRRN(nn.Module):
    def __init__(self):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(25):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out


sr_dir = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir'
sr_error_corr_dir = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_opt_dir'

fluids_files_sr = os.listdir(sr_dir)
fluids_files_sr_sorted = sorted(fluids_files_sr, key=lambda name: int(os.path.splitext(name.split('_')[1])[0]),
                                reverse=False)

progression = 1
File_num = int(len(fluids_files_sr_sorted))
print('The number of files for the correction calculation is {}'.format(File_num))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DRRN().to(device)
# model = error_calibration_module(input_channels=3, num_channels=64, deeper=False).to(device)
# model = error_calibration_module_large_kernel(input_channels=3, num_channels=64).to(device)
model.load_state_dict(torch.load('./Error_correction_models/error_correction_model_finished.pth'))
model.eval()


for npy_file in tqdm(fluids_files_sr_sorted):
    file_path = sr_dir + '/' + npy_file
    fluids_sr_3D = np.load(file_path).astype(np.float32)

    Num_tag = os.path.splitext(npy_file.split('_')[1])[0]

    max_value = np.max(fluids_sr_3D, axis=(0, 1))  # One-dimensional arrays
    min_value = np.min(fluids_sr_3D, axis=(0, 1))
    fluids_sr_3D_01 = imgproc.per_image_normalization(fluids_sr_3D)

    fluids_sr_3D_gpu = torch.from_numpy(fluids_sr_3D_01).to(device=device, dtype=torch.float32)

    fluids_sr_3D_gpu_CHW = fluids_sr_3D_gpu.permute(2, 0, 1).unsqueeze(0)

    fluids_sr_3D_gpu_CHW_error_corr = model(fluids_sr_3D_gpu_CHW)

    fluids_sr_3D_gpu_HWC_error_corr = fluids_sr_3D_gpu_CHW_error_corr.squeeze(0).permute(1, 2, 0)

    fluids_sr_3D_error_corr = fluids_sr_3D_gpu_HWC_error_corr.cpu().detach().numpy()

    fluids_sr_3D_error_corr = fluids_sr_3D_error_corr * (max_value - min_value) + min_value

    np.save(sr_error_corr_dir + '/' + 'SRerrorcorr_' + Num_tag + '.npy', fluids_sr_3D_error_corr)

