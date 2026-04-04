import numpy as np
import torch
import torch.nn as nn

from src.metrics.base_metric import BaseMetric


class STFTMag(nn.Module):
    def __init__(self, nfft=2048, hop=512):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(nfft), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        # x: [B, T]
        stft = torch.stft(
            x,
            self.nfft,
            self.hop,
            window=self.window.to(x.device),
            return_complex=False,
        )
        mag = torch.norm(stft, p=2, dim=-1)
        return mag


class LSD(BaseMetric):
    def __init__(self, nfft=2048, hop=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = kwargs.get("name", "LSD")
        self.stft = STFTMag(nfft=nfft, hop=hop)

    def __call__(self, source=None, predict=None, **kwargs):
        source = source.squeeze(1) if source.dim() == 3 else source
        predict = predict.squeeze(1) if predict.dim() == 3 else predict

        sp = torch.log10(self.stft(source).square().clamp_min(1e-8))
        st = torch.log10(self.stft(predict).square().clamp_min(1e-8))

        val = (sp - st).square().mean(dim=1).sqrt().mean()
        return float(val.detach().cpu().item())


class LSD_LF(BaseMetric):
    def __init__(self, nfft=2048, hop=512, cutoff_freq=4000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = kwargs.get("name", "LSD_LF")
        self.cutoff_freq = cutoff_freq
        self.stft = STFTMag(nfft=nfft, hop=hop)

    def __call__(self, source=None, predict=None, initial_sr=None, target_sr=None, **kwargs):
        source = source.squeeze(1) if source.dim() == 3 else source
        predict = predict.squeeze(1) if predict.dim() == 3 else predict

        sp = torch.log10(self.stft(source).square().clamp_min(1e-8))
        st = torch.log10(self.stft(predict).square().clamp_min(1e-8))

        n_bins = sp.shape[1]
        nyquist = target_sr / 2
        cutoff_bin = int((self.cutoff_freq / nyquist) * n_bins)
        cutoff_bin = max(1, min(cutoff_bin, n_bins))

        val = (sp[:, :cutoff_bin, :] - st[:, :cutoff_bin, :]).square().mean(dim=1).sqrt().mean()
        return float(val.detach().cpu().item())


class LSD_HF(BaseMetric):
    def __init__(self, nfft=2048, hop=512, cutoff_freq=4000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = kwargs.get("name", "LSD_HF")
        self.cutoff_freq = cutoff_freq
        self.stft = STFTMag(nfft=nfft, hop=hop)

    def __call__(self, source=None, predict=None, initial_sr=None, target_sr=None, **kwargs):
        source = source.squeeze(1) if source.dim() == 3 else source
        predict = predict.squeeze(1) if predict.dim() == 3 else predict

        sp = torch.log10(self.stft(source).square().clamp_min(1e-8))
        st = torch.log10(self.stft(predict).square().clamp_min(1e-8))

        n_bins = sp.shape[1]
        nyquist = target_sr / 2
        cutoff_bin = int((self.cutoff_freq / nyquist) * n_bins)
        cutoff_bin = max(0, min(cutoff_bin, n_bins))

        if cutoff_bin >= n_bins:
            return 0.0

        val = (sp[:, cutoff_bin:, :] - st[:, cutoff_bin:, :]).square().mean(dim=1).sqrt().mean()
        return float(val.detach().cpu().item())