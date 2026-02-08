from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.hifi_utils import mel_spectrogram

def _squeeze_ch1(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.size(1) == 1:
        return x.squeeze(1)
    return x

def _norm_to_unit(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    x = x.float()
    mx = x.abs().amax(dim=-1, keepdim=True)
    x = x / (mx + eps)
    return torch.clamp(x, -1.0, 1.0)

def _is_silent(x: torch.Tensor, thr: float = 1e-4) -> torch.Tensor:
    return (x.abs().mean(dim=-1) < thr)


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        batch["pred_audio"] = self.model(batch["audio"])

        T = min(batch["pred_audio"].shape[-1], batch["tg_audio"].shape[-1])
        batch["pred_audio"] = batch["pred_audio"][..., :T]
        batch["tg_audio"]   = batch["tg_audio"][..., :T]

        batch["loss"] = self.criterion(batch["pred_audio"], batch["tg_audio"])

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        pred = _squeeze_ch1(batch["pred_audio"])
        tgt  = _squeeze_ch1(batch["tg_audio"])
        batch["predict"] = pred
        batch["source"]  = tgt

        metrics.update("loss", float(batch["loss"].item()))

        if not self.is_train:
            for met in metric_funcs:
                try:
                    if met.name.upper() == "PESQ":
                        src = _norm_to_unit(batch["source"].detach()).cpu()
                        prd = _norm_to_unit(batch["predict"].detach()).cpu()

                        silent = _is_silent(src) | _is_silent(prd)
                        if silent.all():
                            metrics.update(met.name, 0.0)
                            continue

                        src = src[~silent]
                        prd = prd[~silent]

                        val = met(source=src, predict=prd)

                    else:
                        if hasattr(met, "metric") and hasattr(met.metric, "to"):
                            met.metric = met.metric.to(batch["predict"].device)

                        val = met(source=batch["source"], predict=batch["predict"])

                    if hasattr(val, "item"):
                        val = float(val.item())
                    metrics.update(met.name, float(val))

                except Exception as e:
                    print(f"[WARNING] Metric {met.name} failed: {e}")
                    metrics.update(met.name, 0.0)

        return batch


    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":
            self.log_audio(**batch)
            self.log_melspec(**batch)

    def log_audio(self, audio, tg_audio, pred_audio, **batch):
    # логнем первый пример
        self.writer.add_audio("audio_in",  audio[0].detach().cpu(), 16000)
        self.writer.add_audio("audio_gt",  tg_audio[0].detach().cpu(), 16000)
        self.writer.add_audio("audio_pred", pred_audio[0].detach().cpu(), 16000)

    def log_melspec(self, tg_audio, pred_audio, **batch):
        # (B,1,T) -> (T)
        gt = tg_audio[0].detach().cpu().squeeze(0)
        pr = pred_audio[0].detach().cpu().squeeze(0)

        gt_mel = mel_spectrogram(gt.unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000)
        pr_mel = mel_spectrogram(pr.unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000)

        self.writer.add_image("mel_gt", plot_spectrogram(gt_mel))
        self.writer.add_image("mel_pred", plot_spectrogram(pr_mel))