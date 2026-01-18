import numpy as np
import torch
import torchaudio
from src.metrics.base_metric import BaseMetric
from src.metrics.metric_nets import Wav2Vec2MOS


class MOSNet(BaseMetric):
    def __init__(self, sr=22050, num_splits=8, *args, **kwargs):
        """
        MOSNet metric wrapper around Wav2Vec2MOS inference model.

        Args:
            sr (int): Sampling rate of input audio.
            num_splits (int): Number of samples per split to average metric.
        """
        super().__init__(*args, **kwargs)
        self.name = "MOSNet"
        self.sr = sr
        self.num_splits = num_splits
        self.mos_net = Wav2Vec2MOS("weights/wave2vec2mos.pth")
        self.device = self.mos_net.device  # автоматически определяется внутри

    def __call__(self, source=None, predict=None, **kwargs):
        """
        Args:
            predict (Tensor): (B, T) or (B, 1, T) predicted audio signals
        Returns:
            MOS score (float)
        """
        if predict is None:
            raise ValueError("predict (audio output) must be provided")

        predict = predict / (predict.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        
        resample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.mos_net.sample_rate).to(self.device)

        predict = [resample(p.to(self.device)).squeeze() for p in predict]

        splits = [
            predict[i : i + self.num_splits]
            for i in range(0, len(predict), self.num_splits)
        ]

        scores = [self.mos_net.calculate(split) for split in splits]

        return float(np.mean(scores))
