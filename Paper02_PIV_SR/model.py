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
from typing import Any

import torch
from torch import Tensor
from torch import nn


__all__ = [
    "RDN",
    "rdn_small_x2", "rdn_small_x3", "rdn_small_x4", "rdn_small_x8",
    "rdn_large_x2", "rdn_large_x3", "rdn_large_x4", "rdn_large_x8",
]


# DenseNet是每一层与前面所有层在channel维度上连接（concat）在一起，实现特征复用。
# rb, 残差块, [channels, H, W]------->[channels + growth_channels, H, W]
# 残差是加计算，DenseNet是拼接，这里
class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, growth_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rb(x)
        # 输入网络的数据是四维特征，[batch,C,H,W]，沿着第二个维度拼接就是channel维度上的拼接
        out = torch.cat([identity, out], 1)

        return out


# RDB, H, W不会变，但是每经过一次rb(总共8次), channels都会增加, 再经历一个LFF操作，将channels压缩到64
# index=0，input_channel = 64, output_channel = 64+64
# index=1，input_channel = 128, output_channel = (64+1*64)+64
# index=2，input_channel = 192, output_channel = (64+2*64)+64
# index=3，input_channel = 256, output_channel = (64+3*64)+64
# index=4，input_channel = 320, output_channel = (64+4*64)+64
# index=5，input_channel = 384, output_channel = (64+5*64)+64
# index=6，input_channel = 448, output_channel = (64+6*64)+64
# index=7，input_channel = 512, output_channel = (64+7*64)+64=576
# LFF，input_channel = 576, output_channel = 64，at last, add
# 对于整个rdb结构，输入[channels, H, W]，输出还是[channels, H, W]，完全不会变
class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int, layers: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualBlock(channels + index * growth_channels, growth_channels))
        # nn.Sequential, 一个顺序容器。模块将按照它们在构造函数中传递的顺序添加到其中
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv2d(channels + layers * growth_channels, channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb(x)  # [channels=576, H, W]
        out = self.local_feature_fusion(out)

        out = torch.add(out, identity)  # 张量相加

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        # nn.PixelShuffle(), 重新排列形状张量中的元素(*, C* r**2, H, W)变为(*, C, H*r, W*r)，r是upscale factor.
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


class RDN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rdb: int = 16,
            num_rb: int = 8,
            growth_channels: int = 64,
            upscale_factor: int = 4,
    ) -> None:
        super(RDN, self).__init__()
        self.num_rdb = num_rdb

        # First layer, kernel_size=3, stride=1, padding=1,
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual Dense Blocks, RDB
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(int(num_rdb * channels), channels, (1, 1), (1, 1), (0, 0)),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

        # Upscale block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.conv2(out1)

        outs = []
        # 之所以不直接trunk(out)串联计算出所有最终结果，而采用trunk[i](out)，是为了利用outs存储记录每一个rdb的输出out，将所有的rdb的out进行cat融合
        for i in range(self.num_rdb):
            out = self.trunk[i](out)  # out是不断更新的，上一个rdb的out作为下一个rdb的输入,trunk列表是一个形似下三角
            outs.append(out)

        out = torch.cat(outs, 1)  # [channels*num_rdb, H, W]，第一个维度是batch_size，所以通道拼接沿着axis=1拼接
        out = self.global_feature_fusion(out)  # [channels, H, W]
        out = torch.add(out1, out)  # [channels, H, W]
        out = self.upsampling(out)  # [channels * upscale_factor * upscale_factor ** math.log(upscale_factor, 2), H, W]
        out = self.conv3(out)  # [3, H, W]

        # out = torch.clamp_(out, 0.0, 1.0)  # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量

        return out


def rdn_small_x2(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=2, **kwargs)

    return model


def rdn_small_x3(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=3, **kwargs)

    return model


def rdn_small_x4(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=4, **kwargs)

    return model


def rdn_small_x8(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=8, **kwargs)

    return model


def rdn_large_x2(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=2, **kwargs)

    return model


def rdn_large_x3(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=3, **kwargs)

    return model


def rdn_large_x4(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=4, **kwargs)

    return model


def rdn_large_x8(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=8, **kwargs)

    return model


def rdn_base_cnn(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=2, num_rb=1, growth_channels=3, upscale_factor=4, **kwargs)

    return model
