"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imgproc

np.random.seed(928)
torch.manual_seed(928)
torch.cuda.manual_seed_all(928)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class customdataset_sr_gt(Dataset):
    def __init__(self, sr_dir, gt_dir, ):
        self.sr_dir = sr_dir
        self.gt_dir = gt_dir
        self.sr_files_sorted = sorted(os.listdir(sr_dir),
                                      key=lambda name: int(os.path.splitext(name.split('_')[1])[0]), reverse=False)
        self.gt_files_sorted = sorted(os.listdir(gt_dir),
                                      key=lambda name: int(os.path.splitext(name.split('_')[1])[0]), reverse=False)

    def __getitem__(self, idx):
        assert len(self.sr_files_sorted) == len(self.gt_files_sorted), 'The number of SR and GT data is not equal.'

        # Extract file numbers without relying on file extensions
        sr_num = int(os.path.splitext(self.sr_files_sorted[idx].split('_')[1])[0])
        gt_num = int(os.path.splitext(self.gt_files_sorted[idx].split('_')[1])[0])
        assert sr_num == gt_num, 'The SR file id num and GT file id num is not equal.'

        sr_result = imgproc.per_image_normalization(np.load(os.path.join(self.sr_dir, self.sr_files_sorted[idx])).astype(np.float32))
        gt_result = imgproc.per_image_normalization(np.load(os.path.join(self.gt_dir, self.gt_files_sorted[idx])).astype(np.float32))

        # Convert numpy array to tensor, and convert the channel dimension to the first dimension, [H,W,C]->[C,H,W]
        sr_result_tensor = torch.from_numpy(np.ascontiguousarray(sr_result)).permute(2, 0, 1).float()
        gt_result_tensor = torch.from_numpy(np.ascontiguousarray(gt_result)).permute(2, 0, 1).float()

        return sr_result_tensor, gt_result_tensor

    def __len__(self):
        return len(self.sr_files_sorted)


class Error_Calibration_module(nn.Module):
    """
    The module for error correction, minimizing the prediction error.
    """

    def __init__(self, input_channels, num_channels, element='use_Res'):
        super(Error_Calibration_module, self).__init__()
        self.element = element
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(input_channels + input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        x0 = X

        x1_0 = self.conv1(X)
        x1_1 = F.relu(x1_0)

        x2_0 = self.conv2(x1_1)
        # x2_1 = self.bn1(x2_0)
        x2_1 = F.relu(x2_0)

        x3_0 = self.conv3(x2_1)
        x3_1 = F.relu(x3_0)  #

        if self.element == 'use_Res':
            Y = x3_1 + x0
            return self.conv5(Y)
        elif self.element == 'use_Cat':
            Y = torch.cat((x3_1, x0), 1)
            return self.conv4(Y)
        else:
            raise ValueError('The element is not supported, please check the element.')


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
        x2 = F.relu(x1)
        x3 = self.conv3(x2)
        x4 = F.relu(x3)
        x5 = self.conv2(x4)

        return x5


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 500
batch_size = 64
learning_rate = 0.1

# Specify your directories
sr_directory = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir'
gt_directory = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir'
# Create an instance of customdataset_sr_gt
custom_dataset = customdataset_sr_gt(sr_dir=sr_directory, gt_dir=gt_directory)
# Create DataLoader
train_dataloader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

# model = error_calibration_module(input_channels=3, num_channels=64, deeper=False).to(device)
model = error_calibration_module_large_kernel(input_channels=3, num_channels=64).to(device)
# model = Error_Calibration_module(input_channels=3, num_channels=16, element='use_Res').to(device)


# Loss and optimizer，use MSE LOSS
criterion = nn.MSELoss(reduction='mean')
criterion_L1 = nn.L1Loss(reduction='mean')
criterion_KL = nn.KLDivLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
print_once = True

total_step = len(train_dataloader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (fluids_sr, fluids_gt) in enumerate(train_dataloader):
        model.train()

        fluids_sr = fluids_sr.to(device)
        fluids_gt = fluids_gt.to(device)

        if print_once:
            print(f'fluids_sr.shape: {fluids_sr.shape}, fluids_gt.shape: {fluids_gt.shape}')
            print_once = False  # set flag to False so the print statement won't be executed again in the loop

        # Forward pass
        outputs = model(fluids_sr)
        # loss = criterion(outputs, fluids_gt)
        loss = criterion_L1(outputs, fluids_gt)
        # loss = criterion_KL(F.log_softmax(outputs, dim=1), F.softmax(fluids_gt, dim=1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Decay learning rate
    if (epoch + 1) % 100 == 0:
        curr_lr /= 10
        update_lr(optimizer, curr_lr)

    # Save the model checkpoint
    if not os.path.exists('./Error_correction_models'):
        os.makedirs('./Error_correction_models')
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), './Error_correction_models/model_{}.ckpt'.format(epoch + 1))

# Save the model
torch.save(model.state_dict(), './Error_correction_models/error_correction_model_finished.pth')

