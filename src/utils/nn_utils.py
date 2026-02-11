import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from librosa.filters import mel as librosa_mel_fn
from src.utils.frepainter_attention import PreNorm, Attention, FeedForward



def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        super(ResBlock2, self).__init__()
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class AddSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.add.add(x, self.net(x))


class ConcatSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


def build_block(
        inner_width,
        block_depth,
        mode: Literal["unet_k3_2d", "waveunet_k5"],
        norm
):
    if mode == "unet_k3_2d":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv2d(
                                inner_width,
                                inner_width,
                                3,
                                padding=1,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    elif mode == "waveunet_k5":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv1d(
                                inner_width,
                                inner_width,
                                5,
                                padding=2,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    else:
        raise NotImplementedError


class MultiScaleResnet(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["waveunet_k5"] = "waveunet_k5",
        norm_type: Literal["weight", "spectral", "id"] = "id"
    ):
        super().__init__()
        norm = dict(
            weight=weight_norm, spectral=spectral_norm, id=lambda x: x
        )[norm_type]
        self.in_width = in_width
        self.out_dims = out_width
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(nn.Conv1d(
                    width, inner_width, scale_factor, scale_factor, 0
                ))
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool1d(scale_factor, scale_factor),
                    norm(nn.Conv1d(width, inner_width, 1)),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv1d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(nn.ConvTranspose1d(
                    inner_width, width, scale_factor, scale_factor, 0
                ))
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv1d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv1d(in_width, block_widths[0], 5, padding=2)),
            net,
            norm(nn.Conv1d(block_widths[0], out_width, 5, padding=2)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        return self.net(x)


class MultiScaleResnet2d(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["unet_k3_2d"] = "unet_k3_2d",
        norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.in_width = in_width
        self.out_dims = out_width
        norm = dict(
            weight=weight_norm, spectral=spectral_norm, id=lambda x: x
        )[norm_type]
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(
                    nn.Conv2d(
                        width, inner_width, scale_factor, scale_factor, 0
                    )
                )
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool2d(scale_factor, scale_factor),
                    norm(
                        nn.Conv2d(width, inner_width, 1),
                    ),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv2d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(
                    nn.ConvTranspose2d(
                        inner_width, width, scale_factor, scale_factor, 0
                    )
                )
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv2d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv2d(in_width, block_widths[0], 3, padding=1)),
            net,
            norm(nn.Conv2d(block_widths[0], out_width, 3, padding=1)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        # padding to across spectral dimension to be divisible by 16
        # (max depth assumed to be 4)
        pad = 16 - x.shape[-2] % 16
        shape = x.shape
        padding = torch.zeros((shape[0], shape[1], pad, shape[3])).to(x)
        x1 = torch.cat((x, padding), dim=-2)
        return self.net(x1)[:, :, : x.shape[2]]


class HiFiGeneratorBackbone(torch.nn.Module):
    def __init__(
            self,
            resblock="2",
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            upsample_initial_channel=128,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            conv_pre_kernel_size=1,
            input_channels=80,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.make_conv_pre(
            input_channels,
            upsample_initial_channel,
            conv_pre_kernel_size
        )

        self.ups = None
        self.resblocks = None
        self.out_channels = self.make_resblocks(
            resblock,
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

    def make_conv_pre(self, input_channels, upsample_initial_channel, kernel_size):
        assert kernel_size % 2 == 1
        self.conv_pre = self.norm(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )
        self.conv_pre.apply(init_weights)

    def make_resblocks(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        resblock = (
            ResBlock1 if resblock == "1" else ResBlock2
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                self.norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        ch = None
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock(ch, k, d, norm_type=self.norm_type)
                )
        self.ups.apply(init_weights)
        return ch

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return x



class HiFiGeneratorBackboneV2(torch.nn.Module):
    def __init__(
            self,
            resblock="2",
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            upsample_initial_channel=128,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            conv_pre_kernel_size=1,
            input_channels=80,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.norm_mag = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_ph = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.make_conv_pre(
            input_channels,
            upsample_initial_channel,
            conv_pre_kernel_size
        )

        self.ups_mag = None
        self.ups_ph = None
        self.resblocks_mag = None
        self.resblocks_ph = None
        self.out_channels_mag, self.out_channels_ph = self.make_resblocks(
            resblock,
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

    def make_conv_pre(self, input_channels, upsample_initial_channel, kernel_size):
        assert kernel_size % 2 == 1
        self.conv_pre_mag = self.norm_mag(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )

        self.conv_pre_ph = self.norm_ph(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )
        self.conv_pre_mag.apply(init_weights)
        self.conv_pre_ph.apply(init_weights)

    def make_resblocks(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        resblock = (
            ResBlock1 if resblock == "1" else ResBlock2
        )

        self.ups_mag = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_mag.append(
                self.norm_mag(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        ch_mag = None
        self.resblocks_mag = nn.ModuleList()
        for i in range(len(self.ups_mag)):
            ch_mag = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks_mag.append(
                    resblock(ch_mag, k, d, norm_type=self.norm_type)
                )
        self.ups_mag.apply(init_weights)


        self.ups_ph = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_ph.append(
                self.norm_ph(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        ch_ph = None
        self.resblocks_ph = nn.ModuleList()
        for i in range(len(self.ups_ph)):
            ch_ph = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks_ph.append(
                    resblock(ch_ph, k, d, norm_type=self.norm_type)
                )
        self.ups_ph.apply(init_weights)
        return ch_mag, ch_ph

    def forward(self, mag, ph):
        mag = self.conv_pre_mag(mag)
        ph = self.conv_pre_ph(ph)

        for i in range(self.num_upsamples):
            mag = F.leaky_relu(mag, LRELU_SLOPE)
            ph = F.leaky_relu(ph, LRELU_SLOPE)
            mag = self.ups_mag[i](mag)
            ph = self.ups_ph[i](ph)
            
            mag_s = None
            ph_s = None
            for j in range(self.num_kernels):
                if mag_s is None:
                    mag_s = self.resblocks_mag[i * self.num_kernels + j](mag)
                else:
                    mag_s += self.resblocks_mag[i * self.num_kernels + j](mag)
                if ph_s is None:
                    ph_s = self.resblocks_ph[i * self.num_kernels + j](ph)
                else:
                    ph_s += self.resblocks_ph[i * self.num_kernels + j](ph)
                
            mag = mag_s / self.num_kernels
            ph = ph_s / self.num_kernels

        mag = F.leaky_relu(mag)
        ph = F.leaky_relu(ph)
        return mag, ph





LRELU_SLOPE = 0.1


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*3)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = self.grn = GRN(3*dim)
        self.pwconv2 = nn.Linear(dim*3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id = None) :
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class APNet_BWE_Model(torch.nn.Module):
    def __init__(self, ConvNeXt_layers, ConvNeXt_channels, n_fft):
        super(APNet_BWE_Model, self).__init__()
        self.adanorm_num_embeddings = None
        layer_scale_init_value =  1 / ConvNeXt_layers

        self.conv_pre_mag = nn.Conv1d(n_fft//2, ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(n_fft//2, ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)

        self.convnext_mag = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(ConvNeXt_layers)
            ]
        )

        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(ConvNeXt_layers)
            ]
        )

        self.norm_post_mag = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(ConvNeXt_channels, n_fft//2)
        self.linear_post_pha_r = nn.Linear(ConvNeXt_channels, n_fft//2)
        self.linear_post_pha_i = nn.Linear(ConvNeXt_channels, n_fft//2)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mag_nb, pha_nb):
        print(mag_nb.shape)
        x_mag = self.conv_pre_mag(mag_nb)
        print('conv pre', x_mag.shape)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)
        print(x_mag.shape)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            print('sum', x_mag.shape)
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag, cond_embedding_id=None)
            print('bloack', x_mag.shape)
            x_pha = conv_block_pha(x_pha, cond_embedding_id=None)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        print(x_mag.shape)
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)
        # print('mag', mag_wb.shape) torch.Size([512, 513, 8])

        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)
        print('pha_wb', pha_wb.shape) #torch.Size([512, 513, 8])


        com_wb = torch.stack((torch.exp(mag_wb)*torch.cos(pha_wb), 
                           torch.exp(mag_wb)*torch.sin(pha_wb)), dim=-1)
        
        print('mag shape', mag_wb.shape)
        return mag_wb, pha_wb, com_wb




class SpectralUNet(nn.Module):
    def __init__(
            self,
            block_widths=(8, 16, 24, 32, 64),
            block_depth=5,
            positional_encoding=True,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.norm_type = norm_type
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]

        self.learnable_mel2linspec = norm(nn.Conv1d(80, 513, 1))

        in_width = int(positional_encoding) + 2

        # out_width could be 1 and self.post_conv_2d could be not used here,
        # but both were left for backward compatibility
        # with the other hypotheses we tested
        out_width = block_widths[0]

        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_width,
            out_width=out_width,
            norm_type=norm_type,
        )

        self.post_conv_2d = nn.Sequential(
            norm(nn.Conv2d(out_width, 1, 1, padding=0)),
        )

        self.post_conv_1d = nn.Sequential(
            norm(nn.Conv1d(513, 128, 1, 1, padding=0)),
        )

        self.mel2lin = None
        self.calculate_mel2lin_matrix()

    def calculate_mel2lin_matrix(self):
        mel_np = librosa_mel_fn(
            sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        slices = [
            (np.where(row)[0].min(), np.where(row)[0].max() + 1)
            for row in mel_np
        ]
        slices = [x[0] for x in slices] + [slices[-1][1]]
        mel2lin = np.zeros([81, 513])
        for i, x1, x2 in zip(range(80), slices[:-1], slices[1:]):
            mel2lin[i, x1: x2 + 1] = np.linspace(1, 0, x2 - x1 + 1)
            mel2lin[i + 1, x1: x2 + 1] = np.linspace(0, 1, x2 - x1 + 1)
        mel2lin = mel2lin[1:]
        mel2lin = torch.from_numpy(mel2lin.T).float()
        self.mel2lin = mel2lin

    def mel2linspec(self, mel):
        return torch.matmul(self.mel2lin.to(mel), mel)

    def forward(self, mel):
        linspec_approx = self.mel2linspec(mel)
        linspec_conv_approx = self.learnable_mel2linspec(mel)
        linspec_conv_approx = linspec_conv_approx.view(
            mel.shape[0], 1, -1, mel.shape[2]
        )
        net_input = linspec_approx.view(
            linspec_approx.shape[0], 1, -1, linspec_approx.shape[2]
        )
        if self.positional_encoding:
            pos_enc = torch.linspace(0, 1, 513)[..., None].expand(
                net_input.shape[0], 1, net_input.shape[2], net_input.shape[3]
            )
            net_input = torch.cat((net_input, pos_enc.to(net_input)), dim=1)
        net_input = torch.cat((net_input, linspec_conv_approx), dim=1)
        out = self.net(net_input)
        out = self.post_conv_2d(out).squeeze(1)
        out = self.post_conv_1d(out)
        return out



# class SpectralUNet2(nn.Module):
#     def __init__(
#             self,
#             block_widths=(8, 16, 24, 32, 64),
#             block_depth=5,
#             positional_encoding=True,
#             norm_type: Literal["weight", "spectral"] = "weight",
#     ):
#         super().__init__()
#         self.positional_encoding = positional_encoding
#         self.norm_type = norm_type
#         norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]

#         in_width = 2  # Теперь у нас два канала: real + imag

#         out_width = block_widths[0]  # U-Net оставляем без изменений

#         # U-Net работает с реальными числами, поэтому вход тоже real-valued
#         self.net = MultiScaleResnet2d(
#             block_widths,
#             block_depth,
#             scale_factor=2,
#             in_width=in_width,
#             out_width=out_width,
#             norm_type=norm_type,
#         )

#         self.post_conv_2d = nn.Sequential(
#             norm(nn.Conv2d(out_width, 2, 1, padding=0)),  # Теперь два канала (реальная и мнимая часть)
#         )

#     def forward(self, stft):
#         """
#         Вход: STFT спектрограмма с return_complex=True (tensor с dtype=torch.complex64)
#         Размерность: [batch, freq_bins, time_frames]
#         """

#         # Разделяем на реальные и мнимые части
#         real = stft.real  # [batch, freq_bins, time_frames]
#         imag = stft.imag  # [batch, freq_bins, time_frames]

#         # Объединяем их в два канала
#         net_input = torch.stack([real, imag], dim=1)  # [batch, 2, freq_bins, time_frames]

#         # Применяем U-Net
#         out = self.net(net_input)  # [batch, out_channels, freq_bins, time_frames]

#         # Финальный сверточный слой возвращает два канала (реальная и мнимая части)
#         out = self.post_conv_2d(out)  # [batch, 2, freq_bins, time_frames]

#         # Разделяем обратно на real + imag
#         # real_out, imag_out = out[:, 0], out[:, 1]

#         # Восстанавливаем комплексное представление
#         # out_complex = torch.complex(real_out, imag_out)  # [batch, freq_bins, time_frames]

#         return out

class SpectralUNet2(nn.Module):
    def __init__(
            self,
            block_widths=(8, 16, 24, 32, 64),
            block_depth=5,
            positional_encoding=True,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.norm_type = norm_type
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]

        in_width = 2  # Теперь у нас два канала: real + imag
        out_width = block_widths[0]  # Первый слой U-Net

        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_width,
            out_width=out_width,
            norm_type=norm_type,
        )

        self.post_conv_2d = nn.Sequential(
            norm(nn.Conv2d(out_width, 2, 1, padding=0)),  # Реальная и мнимая часть
        )

        self.post_conv_1d_mag = nn.Sequential(
            norm(nn.Conv1d(513, 128, 1, 1, padding=0)),  # Преобразование 513 → 128
        )

        self.post_conv_1d_ph = nn.Sequential(
            norm(nn.Conv1d(513, 128, 1, 1, padding=0)),  # Преобразование 513 → 128
        )

    def forward(self, stft):
        """
        Вход: STFT спектрограмма с return_complex=True (tensor с dtype=torch.complex64)
        Размерность: [batch, freq_bins, time_frames]
        """

        real = stft.real  # [batch, freq_bins, time_frames]
        imag = stft.imag  # [batch, freq_bins, time_frames]
        net_input = torch.stack([real, imag], dim=1)  # [batch, 2, freq_bins, time_frames]
        out = self.net(net_input)  # [batch, out_channels, freq_bins, time_frames]
        out = self.post_conv_2d(out)  # [batch, 2, freq_bins, time_frames]

        real_out, imag_out = out[:, 0], out[:, 1]

        magnitude = torch.sqrt(real_out**2 + imag_out**2)  # [batch, freq_bins, time_frames]
        phase = torch.atan2(imag_out, real_out)

        out_magnitude = self.post_conv_1d_mag(magnitude)  # [batch, 128, time_frames]
# 
        out_phase = self.post_conv_1d_ph(phase)

        return out_magnitude, out_phase  



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SpectralAttentionUNet(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            depth=2,
            positional_encoding=True,
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        norm = dict(weight=weight_norm, spectral=spectral_norm)['weight']
        self.learnable_mel2linspec = norm(nn.Conv1d(80, 513, 1))

        self.encoder = nn.ModuleList([Transformer(embed_dim, 1, num_heads, embed_dim // num_heads, 4 * embed_dim) for _ in range(depth)]) 

        self.post_conv_1d = nn.Sequential(
            norm(nn.Conv1d(self.embed_dim, 128, 1, 1, padding=0)),
        )
        self.mel2lin = None
        self.calculate_mel2lin_matrix()

    def calculate_mel2lin_matrix(self):
        mel_np = librosa_mel_fn(
            sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        slices = [
            (np.where(row)[0].min(), np.where(row)[0].max() + 1)
            for row in mel_np
        ]
        slices = [x[0] for x in slices] + [slices[-1][1]]
        mel2lin = np.zeros([81, 513])
        for i, x1, x2 in zip(range(80), slices[:-1], slices[1:]):
            mel2lin[i, x1: x2 + 1] = np.linspace(1, 0, x2 - x1 + 1)
            mel2lin[i + 1, x1: x2 + 1] = np.linspace(0, 1, x2 - x1 + 1)
        mel2lin = mel2lin[1:]
        mel2lin = torch.from_numpy(mel2lin.T).float()
        self.mel2lin = mel2lin
        

    def mel2linspec(self, mel):
        print('self.mel2lin', self.mel2lin.shape)
        # torch.Size([513, 80])
        return torch.matmul(self.mel2lin.to(mel), mel)

    def forward(self, mel):
        print('initial shape', mel.shape)
        # initial shape torch.Size([4, 80, 256])
        linspec_approx = self.mel2linspec(mel)
        print('linspec_approx', linspec_approx.shape)
        # torch.Size([4, 513, 256])

        linspec_conv_approx = self.learnable_mel2linspec(mel)
        linspec_conv_approx = linspec_conv_approx.view(
            mel.shape[0], 1, -1, mel.shape[2]
        )

        net_input = linspec_approx.view(
            linspec_approx.shape[0], 1, -1, linspec_approx.shape[2]
        )
        if self.positional_encoding:
            pos_enc = torch.linspace(0, 1, 513)[..., None].expand(
                net_input.shape[0], 1, net_input.shape[2], net_input.shape[3]
            )
            net_input = torch.cat((net_input, pos_enc.to(net_input)), dim=1)
        net_input = torch.cat((net_input, linspec_conv_approx), dim=1)
        print('net_input', net_input.shape)
        #net_input torch.Size([4, 3, 513, 256])

        
        batch_size, num_channels, freq_bins, seq_len = net_input.shape
        
        net_input = net_input.view(batch_size, seq_len, num_channels * freq_bins)
        print('net_input after view', net_input.shape)
        # torch.Size([4, 256, 1539])
        if net_input.size(-1) != self.embed_dim:
            net_input = nn.Linear(net_input.size(-1), self.embed_dim)(net_input)
            print('net_input after linear', net_input.shape)
            # torch.Size([4, 256, 512])
        linspec_embed = net_input
    
        for layer in self.encoder:
            linspec_embed = layer(linspec_embed)
        print('linspec_embed', linspec_embed.shape)
        # linspec_embed = self.decoder(linspec_embed)
        # print('linspec_embed', linspec_embed.shape)
        # torch.Size([4, 256, 512])
        # out = self.output_layer(linspec_embed.transpose(1, 2)).transpose(1, 2)
        out = self.post_conv_1d(linspec_embed.transpose(1, 2))
        print('out', out.shape)

        return out


class SpectralMaskNet(nn.Module):
    def __init__(
        self,
        in_ch=8,
        act="softplus",
        block_widths=(8, 12, 24, 32),
        block_depth=1,
        norm_type: Literal["weight", "spectral", "id"] = "id"
    ):
        super().__init__()
        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_ch,
            out_width=in_ch,
            norm_type=norm_type
        )
        if act == "softplus":
            self.act = nn.Softplus()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        n_fft = 1024
        win_length = n_fft
        hop_length = n_fft // 4
        f_hat = torch.stft(
            x.view(x.shape[0] * x.shape[1], -1),
            n_fft=n_fft,
            center=True,
            hop_length=hop_length,
            window=torch.hann_window(
                window_length=win_length, device=x.device
            ),
            return_complex=False,
        )

        f = (f_hat[:, 1:, 1:].pow(2).sum(-1) + 1e-9).sqrt()

        padding = (
            int(math.ceil(f.shape[-1] / 8.0)) * 8 - f.shape[-1]
        )  # (2**(int(math.ceil(math.log2(f.shape[-1])))) - f.shape[-1]) // 2
        padding_right = padding // 2
        padding_left = padding - padding_right
        f = torch.nn.functional.pad(f, (padding_left, padding_right))

        mult_factor = self.act(
            self.net(f.view(x.shape[0], -1, f.shape[1], f.shape[2]))
        )  # [..., padding_left:-padding_right]
        if padding_right != 0:
            mult_factor = mult_factor[..., padding_left:-padding_right]
        else:
            mult_factor = mult_factor[..., padding_left:]

        mult_factor = mult_factor.reshape(
            (
                mult_factor.shape[0] * mult_factor.shape[1],
                mult_factor.shape[2],
                mult_factor.shape[3],
            )
        )[..., None]

        one_padded_mult_factor = torch.ones_like(f_hat)
        one_padded_mult_factor[:, 1:, 1:] *= mult_factor

        f_hat = torch.view_as_complex(f_hat * one_padded_mult_factor)
        y = torch.istft(
            f_hat,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(
                window_length=win_length, device=x.device
            ),
        )
        return y.view(x.shape[0], x.shape[1], -1)