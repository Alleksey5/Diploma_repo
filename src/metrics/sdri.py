import torch
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SI_SDR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies SI-SDR metric function.
        """
        self.metric = ScaleInvariantSignalDistortionRatio()
        self.use_pit = kwargs.get("use_pit", False)
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
        Returns:
            metric (Tensor): calculated SI-SDR.
        """
        if self.use_pit and self.metric.device != source.device:
            self.metric = self.metric.to(source.device)
        return self.metric(predict, source).mean()