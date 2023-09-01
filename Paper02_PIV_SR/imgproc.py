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
import math
import random
from typing import Any, Tuple

import math
import sys
import time
import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "image_resize", "preprocess_one_image",
    "expand_y", "rgb_to_ycbcr", "bgr_to_ycbcr", "ycbcr_to_bgr", "ycbcr_to_rgb",
    "rgb_to_ycbcr_torch", "bgr_to_ycbcr_torch",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
# 插值核
def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation

    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    # tensor1.type_as(tensor2)将tensor1的数据类型转换为tensor2的数据类型
    # 分段函数计算，(absx <= 1)与((absx > 1) * (absx <= 2))返回的是布尔值True或False，
    # True * False = 0，True * True = 1，True = 1, False = 0
    # Python把0、空字符串''和None看成 False，其他数值和非空字符串都看成 True，
    # x <= 1，计算结果为：(1.5 * absx3 - 2.5 * absx2 + 1)
    # (x > 1) * (x <= 2)，计算结果为：(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2)
    # x在其余范围内，结算结果，直接为0
    # 这里是自定义双三次插值的内核，这个内核可以修改，现在用的是matlab默认的内核。a=-0.5
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:
       weights, indices, sym_len_s, sym_len_e

    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output space maps to 1.5 in input space.
    # 计算逆映射，使输出空间中的0.5映射到输入空间中的1.5，输出空间中0.5+比例映射到输入区域中的1.5。
    # scale定后，就是一个线性map，u----->f(scale)------->x
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the computation?
    # Note: it's OK to use an extra pixel here; if the corresponding weights are all zero, it will be eliminated at the end of this function.
    # math.ceil：向上取整
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output pixel are in row k of the indices matrix.
    # 返回一个二维索引向量，shape:[out_length, p]
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    # math.isclose() 方法返回用于检查两个值是否彼此接近，如果值接近，则返回 True，否则返回 False
    # math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
    # abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    # https://blog.csdn.net/kdongyi/article/details/108180250
    # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
    # 假如weights = torch.transpose(indices)，除了transpose(), 类似的操作还有narrow(), view(), expand()
    # 不会由于weights是contiguous的，weights指向indices，修改weights的同时也会修改indices
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    # np.ascontiguousarray()函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()
    # 删除batch维度,调回(H, W, C),点乘255,截断到[0, 255],转换为uint8类型
    # image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return image


def preprocess_one_image(image_path: str, device: torch.device) -> Tensor:
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # 使用cv2.imread(url)得到的图片是BGR格式，而不是传统的RGB格式，有时需要转换。
    # BGR to RGB，数组形状不变，只是换了显示格式，shape=(h, w, c)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image data to pytorch format data
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def preprocess_one_data(image_path: str, device: torch.device) -> Tuple[Any, ndarray, ndarray]:
    origin_data = np.load(image_path)
    max_value = np.max(origin_data, axis=(0, 1))  # One-dimensional arrays
    min_value = np.min(origin_data, axis=(0, 1))
    # image = cv2.imread(image_path).astype(np.float32) / 255.0
    image = per_image_normalization(origin_data).astype(np.float32)

    # Convert image data to pytorch format data
    # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor, max_value, min_value


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round

    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # 将通道数维度，调到最前
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)  # pytorch的起升维的作用，0表示在张量最外层加一个中括号变成第一维
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    # process H dimension
    # symmetric copying，对称复制
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)  # 32位，“全0”张量
    img_aug.narrow(1, sym_len_hs, in_h).copy_(
        image)  # 在in_c个通道上都进行img_aug的[in_c, sym_len_hs:sym_len_hs+in_h, in_w]的部分被image替换的操作

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1,
                           -1).long()  # torch.arange(start=0, end, step) → 一维Tensor，tensor[sym_patch.size(1)-1,sym_patch.size(1)-2,...0]
    sym_patch_inv = sym_patch.index_select(1,
                                           inv_idx)  # index_select索引函数，1是维度，inv_idx是索引号。sym_patch_inv就是sym_patch在维度1上翻转对称
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    # 至此，img_aug三部分分别赋值完成，其主体由image全部或image各部分赋值构成

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            # 索引出一个二维矩阵------>转置-------->和一个权重向量运算
            # img_aug, shape, [in_c, sym_len_hs + in_h + sym_len_he, in_w]
            # out_1,   shape, [in_c, out_h, in_w]
            # indices, shape, [out_length, p-2]
            # weights, shape, [out_length, p-2]
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    # ==================squeeze_flag是 False=============================
    if squeeze_flag:
        out_2 = out_2.squeeze(0)  # 指定维度上，若该维度的维数为1，则删除该指定维度。（删除一个张量中所有维数为1的维度）
    if numpy_type:
        out_2 = out_2.numpy()  # tensor转numpy数组；https://xujinzh.github.io/2022/03/25/pytorch-tensor-to-numpy-array-and-back/
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


# Bicubic
# 输入的维度必须是三维，二维数组要升维
# 这个是不改变原始值的
# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0


# Paddnig
def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img

    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]
    # Pad the missing eight points

    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]
    return zimg


# https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


# Bicubic operation
def bicubic(img, ratio, a, squeeze_flag=False):
    # Get image size
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape

    img = padding(img, H, W, C)
    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1 / ratio

    # print('Start bicubic interpolation')
    # print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

                # Print progress
                inc = inc + 1
                # sys.stderr.write('\r\033[K' + get_progressbar_str(inc/(C*dH*dW)))
                sys.stderr.flush()
    # sys.stderr.write('\n')
    sys.stderr.flush()
    if squeeze_flag:
        dst = np.transpose(dst, axes=(2, 0, 1))
    return dst


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format,
    and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC form

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr_to_ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


def rgb_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def bgr_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def ycbcr_to_bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[65.481, -37.797, 112.0],
                         [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of bgr2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[24.966, 112.0, -18.214],
                         [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(image: np.ndarray, image_size_H: int, image_size_W: int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image_size_H: The H size of the captured image area.
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size_W (int): The W size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size_H)  # python：[low, high]; numpy：[low, high)，这里用python的
    left = random.randint(0, image_width - image_size_W)
    # 先进行random.randint随机操作，才进入到config文件，固定随机seed，开始训练，不影响这里的随机操作
    # print('随机数为 top={}, left={}'.format(top, left))

    # Crop image patch
    patch_image = image[top:top + image_size_H, left:left + image_size_W, ...]

    return patch_image


def random_crop_pro(lr_image: np.ndarray, hr_image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image.随机裁剪出patch

    Args:
        lr_image (np.ndarray): The input low-resolution image for `OpenCV.imread`.
        hr_image (np.ndarray): The input high-resolution image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        np.ndarray: Small patch images.
    """

    image_height_lr, image_width_lr = lr_image.shape[:2]
    image_height_hr, image_width_hr = hr_image.shape[:2]
    image_height, image_width = image_height_hr, image_width_hr  # 假定hr的尺寸是小于lr的尺寸
    if image_height_lr + image_width_lr <= image_height_hr + image_width_hr:
        image_height, image_width = image_height_lr, image_width_lr

    # Just need to find the top and left coordinates of the image
    # random.randint，返回一个 start 到 end 之间的数字，闭区间
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_lr_image = lr_image[top:top + image_size, left:left + image_size, ...]
    patch_hr_image = hr_image[top:top + image_size, left:left + image_size, ...]

    return patch_lr_image, patch_hr_image


def crop_divide_exactly(data: np.ndarray) -> np.ndarray:
    """
    将原始的PIV数据变形为能够被2，4，8整除的尺寸

    Args:
        data (np.ndarray): The input low-resolution image for `OpenCV.imread`.

    Returns:
        np.ndarray: Small patch images.
    """
    """
    data_fill = np.zeros((84, 32, 3), dtype=np.float32)

    for c in range(3):
        data_fill[1:83, 1:30, c:c + 1] = data[:, :, c:c + 1]
    """
    data_fill = data[1:data.shape[0] - 1, 1:, :]
    return data_fill


def random_rotate(image,
                  angles: list,
                  center: Tuple[int, int] = None,
                  scale_factor: float = 1.0) -> np.ndarray:
    """Rotate an image by a random angle

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image


# code reference: https://discuss.pytorch.org/t/per-image-normalization/22141/4
# tf.image.per_image_standardization: https://www.w3cschool.cn/tensorflow_python/tensorflow_python-vbyx2qyr.html
def per_image_standardization(image):
    """
    This function creates a custom per image standardization transform which is used for data augmentation.
    This function is equivalent to the function tf.image.per_image_standardization in tensorflow.
    params:
        - image (numpy array): Image Array that needs to be standardized.

    returns:
        - image (numpy array): Image Array post standardization.
    """
    # get original data type
    orig_dtype = image.dtype

    # compute image mean
    image_mean = np.mean(image)

    # compute image standard deviation
    stddev = np.std(image)

    # compute number of pixels
    num_pixels = np.array(image.size, dtype=np.float32)

    # compute minimum standard deviation
    min_stddev = np.divide(1., np.sqrt(num_pixels))

    # compute adjusted standard deviation
    adjusted_stddev = np.max([stddev, min_stddev])

    # normalize image
    image -= image_mean
    image = np.divide(image, adjusted_stddev)

    # make sure that image output dtype  == input dtype
    assert image.dtype == orig_dtype

    return image.astype(np.float32)


def per_image_normalization(image):
    """
    This function creates a custom per image normalization transform which is used for data augmentation.

    params:
        - image (numpy array): Image Array that needs to be normalization.

    returns:
        - image (numpy array): Image Array post normalization.
    """
    # get original data type
    orig_dtype = image.dtype

    # compute image max
    image_max = np.max(image, axis=(0, 1))

    # compute image min
    image_min = np.min(image, axis=(0, 1))

    # normalize image
    image -= image_min
    # image = np.divide(image, (image_max - image_min))
    image = image / (image_max - image_min)
    # make sure that image output dtype  == input dtype
    # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    assert image.dtype == orig_dtype

    return image.astype(np.float32)


