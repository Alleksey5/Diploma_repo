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
            x, self.nfft, self.hop,
            window=self.window.to(x.device),
            return_complex=True
        )
        return stft.abs()  # [B, F, frames]


class LSD(BaseMetric):
    """
    Log Spectral Distance (меньше = лучше)
    """
    def __init__(self, nfft=2048, hop=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stft = STFTMag(nfft=nfft, hop=hop)

    def __call__(self, source, predict, **kwargs):
        # source/predict: [B,1,T] или [B,T]
        if source.dim() == 3:
            source = source.squeeze(1)
        if predict.dim() == 3:
            predict = predict.squeeze(1)

        sp = torch.log10(self.stft(source).pow(2).clamp_min(1e-8))
        st = torch.log10(self.stft(predict).pow(2).clamp_min(1e-8))

        lsd = (sp - st).pow(2).mean(dim=1).sqrt().mean()
        return lsd
