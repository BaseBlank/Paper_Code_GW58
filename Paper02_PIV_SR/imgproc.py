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
import random
from typing import Any, Tuple
import math
import cv2
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "imresize", "preprocess_one_image",
    "expand_y", "rgb_to_ycbcr", "bgr_to_ycbcr", "ycbcr_to_bgr", "ycbcr_to_rgb",
    "rgb_to_ycbcr_torch", "bgr_to_ycbcr_torch",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


# reference: stackoverflow, how to perform max/mean pooling on a 2d array using numpy
# reference: https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
def pooling(mat, ksize, method='max', pad=False):
    """
    Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def asStride(arr, sub_shape, stride):
    """
    Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    """
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, method='max', pad=False):
    """
    Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not.
           If no pad, output has size (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    view = asStride(mat_pad, ksize, stride)

    if method == 'max':
        result = np.nanmax(view, axis=(2, 3))
    else:
        result = np.nanmean(view, axis=(2, 3))

    return result


def poolingOverlap3D(arr3d, ksize, stride=None, method='max', pad=False):
    """
    Perform overlapping pooling on a 3D numpy array.

    Args:
        arr3d (numpy.ndarray): 3D input array with shape (3, m, n).
        ksize (tuple): Kernel size in (ky, kx).
        stride (tuple, optional): Stride of pooling window. If None, same as ksize (non-overlapping pooling).
        method (str, optional): Pooling method, 'max' for max-pooling, 'mean' for mean-pooling.
        pad (bool, optional): Whether to pad the input array or not.

    Returns:
        numpy.ndarray: 3D output array with shape (3, m', n'), where m' and n' depend on the pooling parameters.
    """
    pooled_2d = []
    for i in range(arr3d.shape[0]):
        pooled_2d.append(poolingOverlap(arr3d[i, :, :], ksize, stride, method, pad))

    return np.stack(pooled_2d, axis=0)


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
# 插值核
def cubic(x):
    """cubic function used for calculate_weights_indices."""
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) *
                                                                                     (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Calculate weights and indices, used for imresize function.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialias
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


@torch.no_grad()
def imresize(img, scale, antialiasing=True):
    """imresize function same as MATLAB.

    It now only supports bicubic.
    The same scale applies for both height and width.

    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), [0, 1] range.
        scale (float): Scale factor. The same scale applies for both height
            and width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
            Default: True.

    Returns:
        Tensor: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if img.ndim == 2:
            img = img.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = img.size()
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale, kernel, kernel_width,
                                                                             antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale, kernel, kernel_width,
                                                                             antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
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

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def imresize3D(img, scale, antialiasing=True):
    """
        Perform overlapping imresize on a 3D numpy array.

        Args:
            img (numpy.ndarray): 3D input array with shape (3, m, n).
            scale (float): Scale factor.
            antialiasing (bool): Whether to apply anti-aliasing when downsampling.
        Returns:
            numpy.ndarray: 3D output array with shape (3, m', n'), where m' and n' depend on the pooling parameters.
    """
    out_2d = []
    for i in range(img.shape[0]):
        out_2d.append(imresize(img[i, :, :], scale, antialiasing))

    return np.stack(out_2d, axis=0)


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
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)  # HWC to NCHW

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
