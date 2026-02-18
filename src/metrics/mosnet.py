import numpy as np
import torch
import torchaudio
from collections import defaultdict

from src.metrics.base_metric import BaseMetric
from src.metrics.metric_nets import Wav2Vec2MOS


class MOSNet(BaseMetric):
    def __init__(self, sr=16000, num_splits=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = kwargs.get("name", "MOSNet")

        self.sr = sr
        self.num_splits = num_splits

        self.mos_net = Wav2Vec2MOS("weights/wave2vec2mos.pth")
        self.device = self.mos_net.device

        self.result = defaultdict(list)

    def __call__(self, source=None, predict=None, **kwargs):
        if predict is None:
            raise ValueError("predict must be provided")

        self.result["mean"].clear()
        self.result["std"].clear()

        predict = predict / (predict.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr,
            new_freq=self.mos_net.sample_rate
        ).to(self.device)

        predict = [resample(p.to(self.device)).squeeze() for p in predict]

        splits = [
            predict[i: i + self.num_splits]
            for i in range(0, len(predict), self.num_splits)
        ]
        scores = [self.mos_net.calculate(split) for split in splits]

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        self.result["mean"].append(mean)
        self.result["std"].append(std)

        return mean
