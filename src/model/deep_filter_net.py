import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DfParams:
    def __init__(
        self,
        sr: int = 16000,
        fft_size: int = 1024,
        hop_size: int = 256,
        nb_erb: int = 32,
        nb_df: int = 96,
        df_order: int = 5,
        df_lookahead: int = 0,
    ):
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.norm_tau = 1.0
        self.lsnr_max = 35
        self.lsnr_min = -15
        self.min_nb_freqs = 2
        self.df_order = df_order
        self.df_lookahead = df_lookahead
        self.pad_mode = "input"


class ModelParams(DfParams):
    def __init__(self):
        super().__init__(
            sr=16000,
            fft_size=1024,
            hop_size=256,
            nb_erb=32,
            nb_df=96,
            df_order=5,
            df_lookahead=0,
        )

        self.conv_lookahead = 0
        self.conv_k_enc = 2
        self.conv_k_dec = 1
        self.conv_ch = 16
        self.conv_width_f = 1
        self.conv_dec_mode = "transposed"
        self.conv_depthwise = True
        self.convt_depthwise = True
        self.emb_hidden_dim = 256
        self.emb_num_layers = 1
        self.df_hidden_dim = 256
        self.df_num_layers = 3
        self.gru_groups = 1
        self.lin_groups = 1
        self.group_shuffle = True
        self.dfop_method = "real_unfold"
        self.mask_pf = False


class FreqUpsample(nn.Module):
    def __init__(self, factor: int, mode="nearest"):
        super().__init__()
        self.f = float(factor)
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=[1.0, self.f], mode=self.mode)

def convkxf(
    in_ch: int,
    out_ch: Optional[int] = None,
    k: int = 1,
    f: int = 3,
    fstride: int = 2,
    lookahead: int = 0,
    batch_norm: bool = False,
    act: nn.Module = nn.ReLU(inplace=True),
    mode="normal",  # must be "normal", "transposed" or "upsample"
    depthwise: bool = True,
    complex_in: bool = False,
):
    bias = batch_norm is False
    assert f % 2 == 1
    stride = 1 if f == 1 else (1, fstride)
    if out_ch is None:
        out_ch = in_ch * 2 if mode == "normal" else in_ch // 2
    fpad = (f - 1) // 2
    convpad = (0, fpad)
    modules = []
    # Manually pad for time axis kernel to not introduce delay
    pad = (0, 0, k - 1 - lookahead, lookahead)
    if any(p > 0 for p in pad):
        modules.append(("pad", nn.ConstantPad2d(pad, 0.0)))
    if depthwise:
        groups = min(in_ch, out_ch)
    else:
        groups = 1
    if in_ch % groups != 0 or out_ch % groups != 0:
        groups = 1
    if complex_in and groups % 2 == 0:
        groups //= 2
    convkwargs = {
        "in_channels": in_ch,
        "out_channels": out_ch,
        "kernel_size": (k, f),
        "stride": stride,
        "groups": groups,
        "bias": bias,
    }
    if mode == "normal":
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))
    elif mode == "transposed":
        # Since pytorch's transposed conv padding does not correspond to the actual padding but
        # rather the padding that was used in the encoder conv, we need to set time axis padding
        # according to k. E.g., this disables the padding for k=2:
        #     dilation - (k - 1) - padding
        #   = 1        - (2 - 1) - 1 = 0; => padding = fpad (=1 for k=2)
        padding = (k - 1, fpad)
        modules.append(
            ("sconvt", nn.ConvTranspose2d(padding=padding, output_padding=convpad, **convkwargs))
        )
    elif mode == "upsample":
        modules.append(("upsample", FreqUpsample(fstride)))
        convkwargs["stride"] = 1
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))
    else:
        raise NotImplementedError()
    if groups > 1:
        modules.append(("1x1conv", nn.Conv2d(out_ch, out_ch, 1, bias=False)))
    if batch_norm:
        modules.append(("norm", nn.BatchNorm2d(out_ch)))
    modules.append(("act", act))
    return nn.Sequential(OrderedDict(modules))

def erb_fb(widths: np.ndarray, sr: int, normalized: bool = True, inverse: bool = False) -> Tensor:
    n_freqs = int(np.sum(widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

    b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1
    # Normalize to constant energy per resulting band
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0)
    return fb


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.clamp_tensor = torch.__version__ > "1.9.0" or torch.__version__ == "1.9.0"
        self.post_filter = post_filter
        self.eps = eps

    def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
        """Post-Filter proposed by Valin et al. [1].

        Args:
            mask (Tensor): Real valued mask, typically of shape [B, C, T, F].
            beta: Global gain factor.
        Refs:
            [1]: Valin et al.: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech.
        """
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        mask_pf = (1 + beta) * mask / (1 + beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))
        return mask_pf

    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
        # spec (real) [B, 1, T, F, 2], F: freq_bins
        # mask (real): [B, 1, T, Fe], Fe: erb_bins
        # atten_lim: [B]
        if not self.training and self.post_filter:
            mask = self.pf(mask)
        if atten_lim is not None:
            # dB to amplitude
            atten_lim = 10 ** (-atten_lim / 20)
            # Greater equal (__ge__) not implemented for TorchVersion.
            if self.clamp_tensor:
                # Supported by torch >= 1.9
                mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
            else:
                m_out = []
                for i in range(atten_lim.shape[0]):
                    m_out.append(mask[i].clamp_min(atten_lim[i].item()))
                mask = torch.stack(m_out, dim=0)
        mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        return spec * mask


class DfOp(nn.Module):
    df_order: Final[int]
    df_bins: Final[int]
    df_lookahead: Final[int]
    freq_bins: Final[int]

    def __init__(
        self,
        df_bins: int,
        df_order: int = 5,
        df_lookahead: int = 0,
        method: str = "complex_strided",
        freq_bins: int = 0,
    ):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
        self.df_lookahead = df_lookahead
        self.freq_bins = freq_bins
        self.set_forward(method)

    def set_forward(self, method: str):
        # All forward methods should be mathematically similar.
        # DeepFilterNet results are obtained with 'real_unfold'.
        forward_methods = {
            "real_loop": self.forward_real_loop,
            "real_strided": self.forward_real_strided,
            "real_unfold": self.forward_real_unfold,
            "complex_strided": self.forward_complex_strided,
            "real_one_step": self.forward_real_no_pad_one_step,
            "real_hidden_state_loop": self.forward_real_hidden_state_loop,
        }
        if method not in forward_methods.keys():
            raise NotImplementedError(
                f"`method` must be one of {forward_methods.keys()}, but got '{method}'"
            )
        if method == "real_hidden_state_loop":
            assert self.freq_bins >= self.df_bins
            self.spec_buf: Tensor
            # Currently only designed for batch size of 1
            self.register_buffer(
                "spec_buf", torch.zeros(1, 1, self.df_order, self.freq_bins, 2), persistent=False
            )
        self.forward = forward_methods[method]

    def forward_real_loop(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version 0: Manual loop over df_order, maybe best for onnx export?
        b, _, t, _, _ = spec.shape
        f = self.df_bins
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )

        spec_f = torch.zeros((b, t, f, 2), device=spec.device)
        for i in range(self.df_order):
            spec_f[..., 0] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 0]
            spec_f[..., 0] -= padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 1]
            spec_f[..., 1] += padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 0]
            spec_f[..., 1] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 1]
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version1: Use as_strided instead of unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        # Complex numbers are not supported by onnx
        re = padded[..., 0] * coefs[..., 0]
        re -= padded[..., 1] * coefs[..., 1]
        im = padded[..., 1] * coefs[..., 0]
        im += padded[..., 0] * coefs[..., 1]
        spec_f = torch.stack((re, im), -1).sum(2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_unfold(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version2: Unfold
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = spec_pad(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        padded = padded.unfold(dimension=1, size=self.df_order, step=1)  # [B, T, F, 2, O]
        padded = padded.permute(0, 1, 4, 2, 3)
        spec_f = torch.empty_like(padded)
        spec_f[..., 0] = padded[..., 0] * coefs[..., 0]  # re1
        spec_f[..., 0] -= padded[..., 1] * coefs[..., 1]  # re2
        spec_f[..., 1] = padded[..., 1] * coefs[..., 0]  # im1
        spec_f[..., 1] += padded[..., 0] * coefs[..., 1]  # im2
        spec_f = spec_f.sum(dim=2)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_complex_strided(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version3: Complex strided; definatly nicest, no permute, no indexing, but complex gradient
        # spec (real) [B, 1, T, F, 2], O: df_order
        # coefs (real) [B, T, O, F, 2]
        # alpha (real) [B, T, 1]
        padded = as_strided(
            spec[..., : self.df_bins, :].squeeze(1), self.df_order, self.df_lookahead, dim=-3
        )
        spec_f = torch.sum(torch.view_as_complex(padded) * torch.view_as_complex(coefs), dim=2)
        spec_f = torch.view_as_real(spec_f)
        return assign_df(spec, spec_f.unsqueeze(1), self.df_bins, alpha)

    def forward_real_no_pad_one_step(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        # Version4: Only viable for onnx handling. `spec` needs external (ring-)buffer handling.
        # Thus, time steps `t` must be equal to `df_order`.

        # spec (real) [B, 1, O, F', 2]
        # coefs (real) [B, 1, O, F, 2]
        assert (
            spec.shape[2] == self.df_order
        ), "This forward method needs spectrogram buffer with `df_order` time steps as input"
        assert coefs.shape[1] == 1, "This forward method is only valid for 1 time step"
        sre, sim = spec[..., : self.df_bins, :].split(1, -1)
        cre, cim = coefs.split(1, -1)
        outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
        outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
        spec_f = torch.stack((outr, outi), dim=-1)
        return assign_df(
            spec[:, :, self.df_order - self.df_lookahead - 1],
            spec_f.unsqueeze(1),
            self.df_bins,
            alpha,
        )

    def forward_real_hidden_state_loop(self, spec: Tensor, coefs: Tensor, alpha: Tensor) -> Tensor:
        # Version5: Designed for onnx export. `spec` buffer handling is done via a torch buffer.

        # spec (real) [B, 1, T, F', 2]
        # coefs (real) [B, T, O, F, 2]
        b, _, t, _, _ = spec.shape
        spec_out = torch.empty((b, 1, t, self.freq_bins, 2), device=spec.device)
        for t in range(spec.shape[2]):
            self.spec_buf = self.spec_buf.roll(-1, dims=2)
            self.spec_buf[:, :, -1] = spec[:, :, t]
            sre, sim = self.spec_buf[..., : self.df_bins, :].split(1, -1)
            cre, cim = coefs[:, t : t + 1].split(1, -1)
            outr = torch.sum(sre * cre - sim * cim, dim=2).squeeze(-1)
            outi = torch.sum(sre * cim + sim * cre, dim=2).squeeze(-1)
            spec_f = torch.stack((outr, outi), dim=-1)
            spec_out[:, :, t] = assign_df(
                self.spec_buf[:, :, self.df_order - self.df_lookahead - 1].unsqueeze(2),
                spec_f.unsqueeze(1),
                self.df_bins,
                alpha[:, t],
            ).squeeze(2)
        return spec_out


def assign_df(spec: Tensor, spec_f: Tensor, df_bins: int, alpha: Optional[Tensor]):
    spec_out = spec.clone()
    if alpha is not None:
        b = spec.shape[0]
        alpha = alpha.view(b, 1, -1, 1, 1)
        spec_out[..., :df_bins, :] = spec_f * alpha + spec[..., :df_bins, :] * (1 - alpha)
    else:
        spec_out[..., :df_bins, :] = spec_f
    return spec_out


def spec_pad(x: Tensor, window_size: int, lookahead: int, dim: int = 0) -> Tensor:
    pad = [0] * x.dim() * 2
    if dim >= 0:
        pad[(x.dim() - dim - 1) * 2] = window_size - lookahead - 1
        pad[(x.dim() - dim - 1) * 2 + 1] = lookahead
    else:
        pad[(-dim - 1) * 2] = window_size - lookahead - 1
        pad[(-dim - 1) * 2 + 1] = lookahead
    return F.pad(x, pad)


def as_strided(x: Tensor, window_size: int, lookahead: int, step: int = 1, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape.insert(dim + 1, window_size)
    x = spec_pad(x, window_size, lookahead, dim=dim)
    # torch.fx workaround
    step = 1
    stride = [x.stride(0), x.stride(1), x.stride(2), x.stride(3)]
    stride.insert(dim, stride[dim] * step)
    return torch.as_strided(x, shape, stride)


class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        assert (self.hidden_size % groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList(
            (nn.GRU(self.input_size, self.hidden_size, **kwargs) for _ in range(groups))
        )

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        if h0 is None:
            dim0, dim1 = input.shape[:2]
            bs = dim0 if self.batch_first else dim1
            h0 = self.get_h0(bs, device=input.device)
        outputs: List[Tensor] = []
        outstates: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                input[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)
        output = torch.cat(outputs, dim=-1)
        h = torch.cat(outstates, dim=0)
        return output, h


class GroupedGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if groups == 1:
            shuffle = False  # Fully connected, no need to shuffle
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus: List[GroupedGRULayer] = nn.ModuleList()  # type: ignore
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grus.append(GroupedGRULayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def get_h0(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size, self.hidden_size),
            device=device,
        )

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(
            dim0, dim1, self.hidden_size * self.num_directions * self.groups, device=input.device
        )
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (
                    input.view(dim0, dim1, -1, self.groups).transpose(2, 3).reshape(dim0, dim1, -1)
                )
            if self.add_outputs:
                output += input
            else:
                output = input
        outstate = torch.cat(outstates, dim=0)
        return output, outstate


class GroupedLinear(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]
    shuffle: Final[bool]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(orig_shape)
            )
        return output

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        k = p.conv_k_enc
        kwargs = {"batch_norm": True, "depthwise": p.conv_depthwise}
        k0 = 1 if k == 1 and p.conv_lookahead == 0 else max(2, k)
        cl = 1 if p.conv_lookahead > 0 else 0
        self.erb_conv0 = convkxf(1, layer_width, k=k0, fstride=1, lookahead=cl, **kwargs)
        cl = 1 if p.conv_lookahead > 1 else 0
        self.erb_conv1 = convkxf(
            layer_width * wf**0, layer_width * wf**1, k=k, lookahead=cl, **kwargs
        )
        cl = 1 if p.conv_lookahead > 2 else 0
        self.erb_conv2 = convkxf(
            layer_width * wf**1, layer_width * wf**2, k=k, lookahead=cl, **kwargs
        )
        self.erb_conv3 = convkxf(layer_width * wf**2, layer_width * wf**2, k=k, fstride=1, **kwargs)
        self.df_conv0 = convkxf(
            2, layer_width, fstride=1, k=k0, lookahead=p.conv_lookahead, **kwargs
        )
        self.df_conv1 = convkxf(layer_width, layer_width * wf**1, k=k, **kwargs)
        self.erb_bins = p.nb_erb
        self.emb_dim = layer_width * p.nb_erb // 4 * wf**2
        self.df_fc_emb = GroupedLinear(
            layer_width * p.nb_df // 2, self.emb_dim, groups=p.lin_groups
        )
        self.emb_out_dim = p.emb_hidden_dim
        self.emb_n_layers = p.emb_num_layers
        self.gru_groups = p.gru_groups
        self.emb_gru = GroupedGRU(
            self.emb_dim,
            self.emb_out_dim,
            num_layers=p.emb_num_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc]
        cemb = c1.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * Fc/4]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * F/4]
        emb = emb + cemb
        emb, _ = self.emb_gru(emb)
        emb = emb.transpose(0, 1)  # [B, T, C * F/4]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_width = layer_width * wf**2
        self.emb_dim = self.emb_width * (p.nb_erb // 4)
        self.fc_emb = nn.Sequential(
            GroupedLinear(
                p.emb_hidden_dim, self.emb_dim, groups=p.lin_groups, shuffle=p.group_shuffle
            ),
            nn.ReLU(inplace=True),
        )
        k = p.conv_k_dec
        kwargs = {"k": k, "batch_norm": True, "depthwise": p.conv_depthwise}
        tkwargs = {
            "k": k,
            "batch_norm": True,
            "depthwise": p.convt_depthwise,
            "mode": p.conv_dec_mode,
        }
        pkwargs = {"k": 1, "f": 1, "batch_norm": True}
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = convkxf(layer_width * wf**2, self.emb_width, **pkwargs)
        self.convt3 = convkxf(self.emb_width, layer_width * wf**2, fstride=1, **kwargs)
        self.conv2p = convkxf(layer_width * wf**2, layer_width * wf**2, **pkwargs)
        self.convt2 = convkxf(layer_width * wf**2, layer_width * wf**1, **tkwargs)
        self.conv1p = convkxf(layer_width * wf**1, layer_width * wf**1, **pkwargs)
        self.convt1 = convkxf(layer_width * wf**1, layer_width * wf**0, **tkwargs)
        self.conv0p = convkxf(layer_width, layer_width, **pkwargs)
        self.conv0_out = convkxf(layer_width, 1, fstride=1, k=k, act=nn.Sigmoid())

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb = self.fc_emb(emb)
        emb = emb.view(b, t, -1, f8).transpose(1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        self.emb_dim = p.emb_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.gru_groups = p.gru_groups

        self.df_convp = convkxf(
            layer_width, self.df_order * 2, k=1, f=1, complex_in=True, batch_norm=True
        )
        self.df_gru = GroupedGRU(
            p.emb_hidden_dim,
            self.df_n_hidden,
            num_layers=self.df_n_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.df_fc_out = nn.Sequential(
            nn.Linear(self.df_n_hidden, self.df_bins * self.df_order * 2), nn.Tanh()
        )
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb.transpose(0, 1))  # [T, B, H], H: df_n_hidden
        c0 = self.df_convp(c0).transpose(1, 2)  # [B, T, O*2, F]
        c = c.transpose(0, 1)  # [B, T, H]
        alpha = self.df_fc_a(c)  # [B, T, 1]
        c = self.df_fc_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_order * 2, self.df_bins)  # [B, T, O*2, F]
        c = c.add(c0).view(b, t, self.df_order, 2, self.df_bins).transpose(3, 4)  # [B, T, O, F, 2]
        return c, alpha


class DfNet(nn.Module):
    def __init__(
        self,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.freq_bins = p.fft_size // 2 + 1
        self.emb_dim = layer_width * p.nb_erb
        self.erb_bins = p.nb_erb
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)

        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.df_dec = DfDecoder()
        self.df_op = DfOp(
            p.nb_df,
            p.df_order,
            p.df_lookahead,
            freq_bins=self.freq_bins,
            method=p.dfop_method,
        )

        self.run_df = run_df
        if not run_df:
            from loguru import logger

            logger.warning("Running without DF")
        self.train_mask = train_mask

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # Not used, take spec modified by mask instead
        atten_lim: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        spec = self.mask(spec, m, atten_lim)
        if self.run_df:
            df_coefs, df_alpha = self.df_dec(emb, c0)
            spec = self.df_op(spec, df_coefs, df_alpha)
        else:
            df_alpha = torch.zeros(spec.shape[0], spec.shape[2], 1, device=spec.device)
        return spec, m, lsnr, df_alpha

class DeepFilterNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()

        self.sr = p.sr
        self.n_fft = p.fft_size
        self.hop_length = p.hop_size
        self.nb_erb = p.nb_erb
        self.nb_df = p.nb_df

        self.register_buffer("window", torch.hann_window(self.n_fft))

        erb_widths = self._make_erb_widths(self.n_fft // 2 + 1, self.nb_erb)
        erb_inv = erb_fb(erb_widths, self.sr, inverse=True)

        self.erb_fb = erb_fb(erb_widths, self.sr, inverse=False)
        self.dfnet = DfNet(erb_inv_fb=erb_inv, run_df=True, train_mask=True)

    def _make_erb_widths(self, n_freqs: int, nb_erb: int) -> np.ndarray:
        widths = np.full(nb_erb, n_freqs // nb_erb, dtype=np.int64)
        widths[: n_freqs % nb_erb] += 1
        return widths

    def _make_features(self, x_complex: Tensor):
        # x_complex: [B, T, F]
        mag2 = x_complex.abs().pow(2)

        erb = torch.matmul(mag2, self.erb_fb.to(x_complex.device, x_complex.real.dtype))
        feat_erb = torch.log(erb.clamp_min(1e-8)).unsqueeze(1)

        spec_df = x_complex[..., : self.nb_df]
        mean = spec_df.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        spec_df = spec_df / mean

        feat_spec = torch.view_as_real(spec_df).unsqueeze(1)
        return feat_erb, feat_spec

    def forward(self, noisy_wav: Tensor) -> Tensor:
        # noisy_wav: [B, 1, T] или [B, T]
        if noisy_wav.dim() == 3:
            noisy_wav = noisy_wav.squeeze(1)

        length = noisy_wav.shape[-1]
        window = self.window.to(noisy_wav.device, noisy_wav.dtype)

        spec_complex = torch.stft(
            noisy_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
        )

        spec_complex = spec_complex.transpose(1, 2)
        spec = torch.view_as_real(spec_complex).unsqueeze(1)

        feat_erb, feat_spec = self._make_features(spec_complex)

        enhanced_spec, mask, lsnr, df_alpha = self.dfnet(
            spec=spec,
            feat_erb=feat_erb,
            feat_spec=feat_spec,
        )

        enhanced_complex = torch.view_as_complex(
            enhanced_spec.squeeze(1).contiguous()
        )  # [B, T, F]

        enhanced_complex = enhanced_complex.transpose(1, 2)

        enhanced_wav = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=length,
            center=True,
        )

        return enhanced_wav.unsqueeze(1)