"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""

"""Realize the function of dataset preparation."""
# ==============================================================================
from typing import Dict
import os
import queue
import threading

import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import core

import imgproc

__all__ = [
    "TrainValidFlowDataset", "TestFlowDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Load the maximum and minimum values saved locally.
flow_max_final = np.loadtxt("data\\minimax\\flow_max_final.txt")
flow_min_final = np.loadtxt("data\\minimax\\flow_min_final.txt")
flow_max_final = flow_max_final[:, np.newaxis, np.newaxis]  # [C,] -> [C,1,1]
flow_min_final = flow_min_final[:, np.newaxis, np.newaxis]


class TrainValidFlowDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        gt_image_size_H (int): Ground-truth resolution flow field size H.
        gt_image_size_W (int): Ground-truth resolution flow field size W.
        upscale_factor (int): flow field upscale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(
            self,
            train_flow_dir: str,
            gt_flow_size_H: int,
            gt_flow_size_W: int,
            upscale_factor: int,
            mode: str,
            random_method: str = 'random',
    ) -> None:
        super(TrainValidFlowDataset, self).__init__()
        self.flow_file_names = [os.path.join(train_flow_dir, flow_file_name) for flow_file_name in
                                os.listdir(train_flow_dir)]
        self.gt_flow_size_H = gt_flow_size_H
        self.gt_flow_size_W = gt_flow_size_W
        self.upscale_factor = upscale_factor
        self.mode = mode
        self.random_method = random_method

    def __getitem__(self, batch_index: int) -> [Dict[str, Tensor], Dict[str, Tensor]]:
        # Read a batch of flow data, [C, H, W]
        global lr_flow_normalize
        gt_flow = np.load(self.flow_file_names[batch_index]).astype(np.float32)
        # gt_flow_normalize = (gt_flow - flow_min_final) / (flow_max_final - flow_min_final)
        gt_flow_normalize = (gt_flow[0:2, :, :] - flow_min_final[0:2, :, :]) / (flow_max_final[0:2, :, :] - flow_min_final[0:2, :, :])

        # Set antialiasing to True according to the default setting of MATLAB's imresize function.
        if self.random_method == 'random':
            local_random_seed = batch_index * 7 + 13  # 使用任意与批次索引相关的值作为本地随机种子
            random.seed(local_random_seed)  # 设置本地随机种子以保证每次迭代的随机选择独立于全局随机种子

            use_pooling_choice = ['pool', 'gaussian', 'cubic', ]
            use_pooling_method = random.choice(use_pooling_choice)

            if use_pooling_method == 'pool':
                lr_flow_normalize = (imgproc.poolingOverlap3D(arr3d=gt_flow_normalize,
                                                              ksize=(self.upscale_factor, self.upscale_factor),
                                                              stride=None,
                                                              method='mean',
                                                              pad=False))
            elif use_pooling_method == 'gaussian':
                gt_flow_normalize_tensor = torch.from_numpy(gt_flow_normalize)
                if cuda.is_available():
                    gt_flow_normalize_tensor = gt_flow_normalize_tensor.cuda()
                lr_flow_normalize_tensor = core.imresize(gt_flow_normalize_tensor,
                                                         scale=1 / self.upscale_factor,
                                                         antialiasing=True,
                                                         kernel='gaussian', sigma=1)
                lr_flow_normalize = lr_flow_normalize_tensor.cpu().numpy()
            elif use_pooling_method == 'cubic':
                gt_flow_normalize_tensor = torch.from_numpy(gt_flow_normalize)
                if cuda.is_available():
                    gt_flow_normalize_tensor = gt_flow_normalize_tensor.cuda()
                lr_flow_normalize_tensor = core.imresize(gt_flow_normalize_tensor,
                                                         scale=1 / self.upscale_factor,
                                                         antialiasing=True,
                                                         kernel='cubic')
                lr_flow_normalize = lr_flow_normalize_tensor.cpu().numpy()
        elif self.random_method == 'pooling':
            lr_flow_normalize = imgproc.poolingOverlap3D(arr3d=gt_flow_normalize,
                                                         ksize=(self.upscale_factor, self.upscale_factor),
                                                         stride=None,
                                                         method='mean',
                                                         pad=False)
        elif self.random_method == 'resize_gaussian':
            gt_flow_normalize_tensor = torch.from_numpy(gt_flow_normalize)
            lr_flow_normalize_tensor = core.imresize(gt_flow_normalize_tensor,
                                                     scale=1 / self.upscale_factor,
                                                     antialiasing=True,
                                                     kernel='gaussian', sigma=1)
            lr_flow_normalize = lr_flow_normalize_tensor.cpu().numpy()

        elif self.random_method == 'resize_cubic':
            gt_flow_normalize_tensor = torch.from_numpy(gt_flow_normalize)
            lr_flow_normalize_tensor = core.imresize(gt_flow_normalize_tensor,
                                                     scale=1 / self.upscale_factor,
                                                     antialiasing=True,
                                                     kernel='cubic')
            lr_flow_normalize = lr_flow_normalize_tensor.cpu().numpy()
        else:
            raise ValueError("Invalid random_method. Options are 'random', 'resize' and 'pooling'.")

        lr_flow_tensor_normalize = torch.from_numpy(np.ascontiguousarray(lr_flow_normalize)).float()  # [C, H, W]
        gt_flow_tensor_normalize = torch.from_numpy(np.ascontiguousarray(gt_flow_normalize)).float()

        return {"gt": gt_flow_tensor_normalize, "lr": lr_flow_tensor_normalize}

    def __len__(self) -> int:
        return len(self.flow_file_names)


class TestFlowDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_flows_dir (str): ground truth flow field in test set
        test_lr_flows_dir (str): low-resolution flow field in test set
    """

    def __init__(self, test_gt_flows_dir: str, test_lr_flows_dir: str) -> None:
        super(TestFlowDataset, self).__init__()
        # Get all flow file names in folder
        self.gt_flow_file_names = [os.path.join(test_gt_flows_dir, x) for x in os.listdir(test_gt_flows_dir)]
        self.lr_flow_file_names = [os.path.join(test_lr_flows_dir, x) for x in os.listdir(test_lr_flows_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of flow data
        gt_flow = np.load(self.gt_flow_file_names[batch_index]).astype(np.float32)
        lr_flow = np.load(self.lr_flow_file_names[batch_index]).astype(np.float32)

        # gt_tensor = torch.from_numpy(np.ascontiguousarray(gt_flow)).float()  # [C, H, W]
        # lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_flow)).float()

        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt_flow[0:2, :, :])).float()  # [C, H, W]
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_flow[0:2, :, :])).float()

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_flow_file_names)


# 原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
# 使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。
# https://blog.csdn.net/Rocky6688/article/details/105317098
class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


# CUDAPrefetcher(train_dataloader, config.device)
class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)  # 采用iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。也可以使用enumerate(dataloader)的形式访问。
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
