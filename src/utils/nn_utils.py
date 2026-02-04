import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from librosa.filters import mel as librosa_mel_fn


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()

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
        y = self.net(x)

        tx = x.shape[-1]
        ty = y.shape[-1]

        if ty < tx:
            y = F.pad(y, (0, tx - ty))
        elif ty > tx:
            y = y[..., :tx]

        return torch.cat([x, y], dim=1)


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
        pad = (16 - (x.shape[-2] % 16)) % 16
        if pad != 0:
            shape = x.shape
            padding = torch.zeros((shape[0], shape[1], pad, shape[3]), device=x.device, dtype=x.dtype)
            x1 = torch.cat((x, padding), dim=-2)
        else:
            x1 = x
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

class SMNLikeERBGainNet(nn.Module):
    def __init__(
        self,
        n_erb_bands: int,
        block_widths=(8, 12, 24, 32),
        block_depth=4,
        norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.n_erb_bands = n_erb_bands

        self.net = MultiScaleResnet2d(
            block_widths=block_widths,
            block_depth=block_depth,
            scale_factor=2,
            in_width=1,
            out_width=1,
            norm_type=norm_type,
            mode="unet_k3_2d",
        )

    def forward(self, erb_feat: torch.Tensor) -> torch.Tensor:
        assert erb_feat.dim() == 3, f"Expected (BC,B,K), got {erb_feat.shape}"
        assert erb_feat.shape[1] == self.n_erb_bands, \
            f"Expected n_erb_bands={self.n_erb_bands}, got {erb_feat.shape[1]}"

        x = erb_feat.unsqueeze(1)
        y = self.net(x)
        return y.squeeze(1)

class DFNetPostProcessor(nn.Module):
    def __init__(
        self,
        in_ch: int,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_erb_bands: int = 32,
        gain_range_db: float = 12.0,
        df_order: int = 3,
        f_df_max: int = 6000,
        enc_hidden: int = 64,
        df_hidden: int = 32,
        use_df: bool = True,
        norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.in_ch = in_ch
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.n_erb_bands = n_erb_bands
        self.gain_range_db = float(gain_range_db)
        self.df_order = int(df_order)
        self.f_df_max = int(f_df_max)
        self.use_df = bool(use_df)
        self.enc_hidden = int(enc_hidden)
        self.df_hidden = int(df_hidden)

        self.gain_net = SMNLikeERBGainNet(
            n_erb_bands=n_erb_bands,
            block_widths=(8, 12, 24, 32),
            block_depth=4,
            norm_type=norm_type,
        )

        self.df_enc = nn.Sequential(
            nn.Conv1d(n_erb_bands, enc_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(enc_hidden, enc_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )

        self.df_time_proj = nn.Sequential(
            nn.Conv1d(enc_hidden, df_hidden, kernel_size=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )

        self._df_head: Optional[nn.Linear] = None
        self._erb_fb: Optional[torch.Tensor] = None
        self._erb_fb_t: Optional[torch.Tensor] = None

    def _build_erb_fb(self, device: torch.device, dtype: torch.dtype, n_freq: int):
        fb = librosa_mel_fn(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_erb_bands,
            fmin=0,
            fmax=self.sr / 2,
        )
        fb = torch.from_numpy(fb).to(device=device, dtype=dtype)

        fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-8)

        fb_t = fb.transpose(0, 1)
        fb_t = fb_t / (fb_t.sum(dim=1, keepdim=True) + 1e-8)

        self._erb_fb = fb
        self._erb_fb_t = fb_t

    def _init_df_head(self, device: torch.device, dtype: torch.dtype, n_freq: int):
        max_bin = int((self.f_df_max / (self.sr / 2)) * (n_freq - 1))
        max_bin = max(0, min(n_freq - 1, max_bin))
        self._df_max_bin = max_bin

        out_dim = (max_bin + 1) * (self.df_order + 1) * 2
        self._df_head = nn.Linear(self.df_hidden, out_dim).to(device=device, dtype=dtype)

    def _stft(self, x_bc_t: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=x_bc_t.device, dtype=x_bc_t.dtype)
        X = torch.stft(
            x_bc_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        return X

    def _istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=X.device, dtype=X.real.dtype)
        y = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            length=length,
        )
        return y

    def _erb_features(self, mag: torch.Tensor) -> torch.Tensor:
        erb = torch.einsum("bf,cfk->cbk", self._erb_fb, mag)
        erb = torch.log1p(erb)
        return erb

    def _erb_to_freq_gains(self, gains_erb: torch.Tensor) -> torch.Tensor:
        gains_freq = torch.einsum("fb,cbk->cfk", self._erb_fb_t, gains_erb)
        return gains_freq

    def _apply_gain_stage(self, X: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(X)

        if self._erb_fb is None or self._erb_fb.device != X.device or self._erb_fb.dtype != mag.dtype:
            self._build_erb_fb(X.device, mag.dtype, n_freq=mag.shape[1])

        erb_feat = self._erb_features(mag)
        gain_logits = self.gain_net(erb_feat)

        gain_db = self.gain_range_db * torch.tanh(gain_logits)
        gain_lin_erb = torch.pow(10.0, gain_db / 20.0)

        gain_lin_freq = self._erb_to_freq_gains(gain_lin_erb)

        Xg = X * gain_lin_freq.to(X.dtype)
        return Xg, erb_feat

    def _apply_df_stage(self, X: torch.Tensor, erb_feat: torch.Tensor) -> torch.Tensor:
        if not self.use_df or self.df_order <= 0:
            return X

        BC, n_freq, K = X.shape

        dtype = X.dtype
        device = X.device

        if self._df_head is None:
            self._init_df_head(device, erb_feat.dtype, n_freq=n_freq)

        h = self.df_enc(erb_feat)
        h = self.df_time_proj(h)

        h_t = h.permute(0, 2, 1).contiguous()
        taps_flat = self._df_head(h_t)

        max_bin = self._df_max_bin
        n_taps = self.df_order + 1

        taps_flat = taps_flat.view(BC, K, (max_bin + 1), n_taps, 2)
        taps = torch.complex(taps_flat[..., 0], taps_flat[..., 1])

        Xdf = X[:, :max_bin + 1, :]
        Xdf_pad = F.pad(Xdf, (self.df_order, 0))
        Xr = Xdf_pad.reshape(BC * (max_bin + 1), 1, K + self.df_order)
        win = Xr.unfold(dimension=2, size=n_taps, step=1)
        win = win.squeeze(1)
        taps2 = taps.permute(0, 2, 1, 3).contiguous()
        taps2 = taps2.view(BC * (max_bin + 1), K, n_taps)

        Ydf = (win * taps2).sum(dim=-1)
        Ydf = Ydf.view(BC, (max_bin + 1), K)

        X_out = X.clone()
        X_out[:, :max_bin + 1, :] = Ydf
        return X_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected (B,C,T), got {x.shape}"
        B, C, T = x.shape
        assert C == self.in_ch, f"DFNetPostProcessor expects C={self.in_ch}, got {C}"

        x_bc = x.reshape(B * C, T)
        X = self._stft(x_bc)
        Xg, erb_feat = self._apply_gain_stage(X)

        Xdf = self._apply_df_stage(Xg, erb_feat)

        y_bc = self._istft(Xdf, length=T)
        y = y_bc.view(B, C, T)
        return y

