"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of GitHub.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""
# ==============================================================================
import os
import time
from typing import List
import numpy as np

import torch
from torch import nn, Tensor
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, TrainValidFlowDataset, TestFlowDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    # train_prefetcher与valid_prefetcher的batch_size由config文件决定，test_prefetcher的batch_size=1
    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    rdn_model, ema_rdn_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(rdn_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        rdn_model = load_state_dict(rdn_model, config.pretrained_model_weights_path)
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # 检查是否恢复了预训练的模型
    print("Check whether the pretrained model is restored...")
    if config.resume_model_weights_path:
        rdn_model, ema_rdn_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            rdn_model,
            config.pretrained_model_weights_path,
            ema_rdn_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)  # 训练过程的模型参数
    results_dir = os.path.join("results", config.exp_name)
    loss_figure_dir = os.path.join("results", 'loss_figure')
    make_directory(samples_dir)
    make_directory(results_dir)
    make_directory(loss_figure_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    # Training loss record drawing
    loss_figure = []
    psrn_figure = []
    ssim_figure = []
    val_loss_figure = []
    for epoch in range(start_epoch, config.epochs):
        loss_record = train(rdn_model, ema_rdn_model, train_prefetcher, criterion, optimizer, epoch, scaler, writer)
        psnr, ssim, val_loss_record = validate(rdn_model, test_prefetcher, epoch, writer, psnr_model, ssim_model,
                                               criterion, "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index，保存训练文件
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": rdn_model.state_dict(),
                         "ema_state_dict": ema_rdn_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "best.pth.tar",
                        "last.pth.tar",
                        is_best,
                        is_last)

        loss_record_np = [i.cpu().detach().numpy() for i in loss_record]
        loss_figure.append(loss_record_np)
        psrn_figure.append(psnr)
        ssim_figure.append(ssim)
        val_loss_record_np = [i.cpu().detach().numpy() for i in val_loss_record]
        val_loss_figure.append(val_loss_record_np)

    try:
        loss_figure = np.array(loss_figure, dtype=np.float32)
        np.savetxt(os.path.join(loss_figure_dir, 'loss_figure.csv'), loss_figure, fmt='%.32f', delimiter=',')
        np.savetxt(os.path.join(loss_figure_dir, 'psrn_figure.csv'), psrn_figure, fmt='%.32f', delimiter=',')
        np.savetxt(os.path.join(loss_figure_dir, 'ssim_figure.csv'), ssim_figure, fmt='%.32f', delimiter=',')
        np.savetxt(os.path.join(loss_figure_dir, 'val_loss_figure.csv'), val_loss_figure, fmt='%.32f', delimiter=',')

    except:
        print('loss_figure.csv file was not saved successfully')


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidFlowDataset(config.train_flow_dir,
                                           config.gt_flow_size_H,
                                           config.gt_flow_size_W,
                                           config.upscale_factor,
                                           "Train",
                                           config.random_method, )
    test_datasets = TestFlowDataset(config.test_gt_flows_dir,
                                    config.test_lr_flows_dir, )

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    rdn_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                       out_channels=config.out_channels,
                                                       channels=config.channels)
    rdn_model = rdn_model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
                                                                                      1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_rdn_model = AveragedModel(rdn_model, avg_fn=ema_avg)

    return rdn_model, ema_rdn_model


# L1Loss不是平方误差，而是绝对值误差
def define_loss() -> nn.L1Loss:
    criterion = nn.L1Loss()
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(rdn_model) -> optim.Adam:
    optimizer = optim.Adam(rdn_model.parameters(),
                           config.model_lr,
                           config.model_betas,
                           config.model_eps,
                           config.model_weight_decay)

    return optimizer


# 随着epoch的增大而逐渐减小学习率，阶梯式衰减，每个一定的epoch，lr会自动乘以gamma进行阶梯式衰减
def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer,
                                    config.lr_scheduler_step_size,
                                    config.lr_scheduler_gamma)
    return scheduler


def train(
        rdn_model: nn.Module,
        ema_rdn_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> List[Tensor]:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    rdn_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    loss_record = []
    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        rdn_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = rdn_model(lr)
            loss = torch.mul(config.loss_weights, criterion(sr, gt))

            loss_record.append(loss)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_rdn_model.update_parameters(rdn_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

    return loss_record


def validate(
        rdn_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        criterion: nn.L1Loss,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    rdn_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    # record val loss
    loss_record = []

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            sr = rdn_model(lr)

            # gain val loss
            loss = torch.mul(config.loss_weights, criterion(sr, gt))
            loss_record.append(loss)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg, loss_record


if __name__ == "__main__":
    main()
