"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of github.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""
# ==============================================================================
from typing import Dict

"""Realize the function of dataset preparation."""
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        gt_image_size_H (int): Ground-truth resolution image size H.
        gt_image_size_W (int): Ground-truth resolution image size W.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(
            self,
            image_dir: str,
            gt_image_size_H: int,
            gt_image_size_W: int,
            upscale_factor: int,
            mode: str,
    ) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        self.gt_image_size_H = gt_image_size_H
        self.gt_image_size_W = gt_image_size_W
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [Dict[str, Tensor], Dict[str, Tensor]]:
        # Read a batch of image data, [H, W, C]
        # gt_image = cv2.imread(self.image_file_names[batch_index]).astype(np.float32) / 255.
        gt_image = imgproc.per_image_normalization(np.load(self.image_file_names[batch_index])).astype(np.float32)
        # Image processing operations
        if self.mode == "Train":
            # 之所以裁剪，是为了将不同缩放尺寸统一patch的大小
            # 因为一个原始图像经过imgproc.imresize(lr_image, 1/upscale_factor)与imresize(lr_image, upscale_factor)后的大小，可能与原始大小差 +-1
            gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size_H, self.gt_image_size_W)
            # gt_crop_image = imgproc.crop_divide_exactly(gt_image)
            # gt_crop_image = imgproc.random_rotate(gt_crop_image, [90, 180, 270])
            # gt_crop_image = imgproc.random_horizontally_flip(gt_crop_image, 0.5)
            # gt_crop_image = imgproc.random_vertically_flip(gt_crop_image, 0.5)
        elif self.mode == "Valid":
            # gt_crop_image = imgproc.crop_divide_exactly(gt_image)
            gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size_H, self.gt_image_size_W)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)  # [H,W,C]
        # lr_crop_image = imgproc.bicubic(gt_crop_image, ratio=1 / self.upscale_factor, a=-0.5, squeeze_flag=False)

        # BGR convert RGB
        # gt_crop_image = cv2.cvtColor(gt_crop_image, cv2.COLOR_BGR2RGB)
        # lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        # image_to_tensor操作后，维度已经由[H,W,C]变为了[C,H,W]
        gt_crop_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)  # [H,W,C]变为了[C,H,W]
        lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)

        return {"gt": gt_crop_tensor, "lr": lr_crop_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str, upscale_factor: int, gt_image_size_H: int,
                 gt_image_size_W: int, ) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.upscale_factor = upscale_factor
        self.gt_image_size_H = gt_image_size_H
        self.gt_image_size_W = gt_image_size_W
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
        # self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        # gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
        gt_image = imgproc.per_image_normalization(np.load(self.gt_image_file_names[batch_index])).astype(np.float32)
        # lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.
        # lr_image = imgproc.per_image_normalization(np.load(self.lr_image_file_names[batch_index])).astype(np.float32)

        # BGR convert RGB
        # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # gt_image = imgproc.crop_divide_exactly(gt_image)
        gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size_H, self.gt_image_size_W)
        lr_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)
        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


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
