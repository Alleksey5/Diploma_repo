from torchmetrics.audio.snr import SignalNoiseRatio

from src.metrics.base_metric import BaseMetric

class SNR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies SNR metric function.
        """
        self.metric = SignalNoiseRatio()
        self.use_pit = kwargs.get("use_pit", False)
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
        Returns:
            metric (Tensor): calculated SNR.
        """
        if self.use_pit and self.metric.device != source.device:
            self.metric = self.metric.to(source.device)
        return self.metric(predict, source).mean()