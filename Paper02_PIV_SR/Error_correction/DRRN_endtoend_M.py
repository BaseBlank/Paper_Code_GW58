"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
"""
# ==============================================================================
import torch
from torch import nn
from math import sqrt
from torch.nn import functional as F
import os
import numpy as np
import math
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
    def __init__(self, sr_dir, gt_dir, upscale_factor=4):
        self.sr_dir = sr_dir
        self.gt_dir = gt_dir
        self.upscale_factor = upscale_factor
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

        sr_result = np.load(os.path.join(self.sr_dir, self.sr_files_sorted[idx])).astype(np.float32)
        gt_result = np.load(os.path.join(self.gt_dir, self.gt_files_sorted[idx])).astype(np.float32)  # [H,W,C]

        # Replace the direction of the third channel with sqrt(u**2+v**2).
        sr_result[:, :, 2] = np.sqrt(sr_result[:, :, 0] ** 2 + sr_result[:, :, 1] ** 2)
        gt_result[:, :, 2] = np.sqrt(gt_result[:, :, 0] ** 2 + gt_result[:, :, 1] ** 2)

        # sr_result = imgproc.per_image_normalization(sr_result)
        # gt_result = imgproc.per_image_normalization(gt_result)  # [H,W,C]

        # Only the tensor sqrt(u**2+v**2) of the last dimension is kept, [H,W,C]->[H,W,1]
        sr_result = sr_result[:, :, 2:3]
        gt_result = gt_result[:, :, 2:3]

        # Convert numpy array to tensor, and convert the channel dimension to the first dimension, [H,W,C]->[C,H,W]
        sr_result_tensor = torch.from_numpy(np.ascontiguousarray(sr_result)).permute(2, 0, 1).float()
        gt_result_tensor = torch.from_numpy(np.ascontiguousarray(gt_result)).permute(2, 0, 1).float()

        return sr_result_tensor, gt_result_tensor

    def __len__(self):
        return len(self.sr_files_sorted)


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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 200
batch_size = 64
learning_rate = 0.1

# Specify your directories
sr_directory = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/sr_dir'
gt_directory = 'F:/model_generator/No_Heating/center_blockage_0.7/test_use/gt_dir'
# Create an instance of customdataset_sr_gt
custom_dataset = customdataset_sr_gt(sr_dir=sr_directory, gt_dir=gt_directory)
# Create DataLoader
train_dataloader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

model = DRRN(num_residual_unit=25).to(device)


# Loss and optimizer，use MSE LOSS
# criterion = nn.L1Loss(reduction='mean')
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)


class RelativeErrorLoss(nn.Module):
    def __init__(self, infinitesimal=1e-8):
        super(RelativeErrorLoss, self).__init__()
        self.infinitesimal = infinitesimal

    def forward(self, predicted, actual):
        # absolute_error = torch.abs(predicted - actual)
        absolute_error = predicted - actual
        # print(f'absolute_error\n: {absolute_error[0, 0, :, :]}')
        # Replace elements with 0 with infinitesimal
        absolute_error[absolute_error == 0] = self.infinitesimal
        # relative_error = absolute_error / torch.abs(actual + self.infinitesimal)  # Adding a small epsilon to avoid division by zero
        relative_error = torch.abs(absolute_error / (actual + self.infinitesimal))
        loss = torch.mean(relative_error)
        return loss


criterion = RelativeErrorLoss()

# print(model)
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.1 * (0.5 ** (epoch // 5))
    return lr


# Train the model
print_once = True

total_step = len(train_dataloader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    # lr policy
    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 2
        update_lr(optimizer, curr_lr)
    print("Epoch={}, lr={}".format(epoch + 1, optimizer.param_groups[0]["lr"]))

    model.train()
    for i, (fluids_sr, fluids_gt) in enumerate(train_dataloader):
        fluids_sr = fluids_sr.to(device)
        fluids_gt = fluids_gt.to(device)

        if print_once:
            print(f'fluids_sr.shape: {fluids_sr.shape}, fluids_gt.shape: {fluids_gt.shape}')
            print_once = False  # set flag to False so the print statement won't be executed again in the loop

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass
        outputs = model(fluids_sr)
        loss = criterion(outputs, fluids_gt)

        # Backward and optimize and Gradient Clipping
        loss.backward()
        clip = 0.01 / curr_lr
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    if not os.path.exists('./Error_correction_models'):
        os.makedirs('./Error_correction_models')
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), './Error_correction_models/DRRN_enetoend_model_{}.ckpt'.format(epoch + 1))

# Save the model
torch.save(model.state_dict(), './Error_correction_models/error_correction_DRRN_enetoend_model_finished.pth')
