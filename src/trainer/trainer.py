# src/trainer/trainer.py
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import clip_grad_norm_

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Supervised trainer for audio super-resolution:
      wav_lr -> model -> wav_pred  (target: wav_hr)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mel params (for logging only)
        self.mel_n_fft = 1024
        self.mel_n_mels = 80
        self.mel_hop = 256
        self.mel_win = 1024
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

    def create_mel_spec(self, wav_1d: torch.Tensor, sr: int) -> torch.Tensor:
        """
        wav_1d: (B,T)
        returns: (B, n_mels, frames)  (log-mel)
        """
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.mel_n_fft,
            win_length=self.mel_win,
            hop_length=self.mel_hop,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.mel_n_mels,
            power=1.0,
            normalized=False,
        ).to(wav_1d.device)

        m = mel(wav_1d)
        m = torch.log(torch.clamp(m, min=1e-5))
        return m

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)

        wav_lr = batch["wav_lr"]  # (B,1,T)
        wav_hr = batch["wav_hr"]  # (B,1,T)

        if self.is_train:
            self.optimizer.zero_grad()

        wav_pred = self.model(wav_lr)  # (B,1,T) ожидается
        batch["wav_pred"] = wav_pred

        batch["generated_wav"] = wav_pred

        # loss
        loss = F.l1_loss(wav_pred, wav_hr)
        batch["loss"] = loss

        if self.is_train:
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", loss.item())

        target_sr = int(batch.get("target_sr", 16000))
        batch["mel_spec_hr"] = self.create_mel_spec(wav_hr.squeeze(1), sr=target_sr).detach()
        batch["mel_spec_fake"] = self.create_mel_spec(wav_pred.squeeze(1), sr=target_sr).detach()

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if self.writer is None:
            return

        part = "train" if mode == "train" else "val"

        # ---- audio logging ----
        wav_lr = batch["wav_lr"]
        wav_hr = batch["wav_hr"]
        generated = batch["generated_wav"]

        initial_sr = int(batch.get("initial_sr", 4000))
        target_sr = int(batch.get("target_sr", 16000))

        n = min(2, wav_lr.shape[0])
        for i in range(n):
            self.writer.add_audio(f"{part}/lr_{i}", wav_lr[i], initial_sr)
            self.writer.add_audio(f"{part}/hr_{i}", wav_hr[i], target_sr)
            self.writer.add_audio(f"{part}/pred_{i}", generated[i], target_sr)

        # ---- spectrogram logging ----
        mel_lr = self.create_mel_spec(wav_lr.squeeze(1), sr=initial_sr)
        mel_hr = batch["mel_spec_hr"]
        mel_fake = batch["mel_spec_fake"]

        for i in range(n):
            self.writer.add_image(f"{part}/melspec_lr_{i}", plot_spectrogram(mel_lr[i].detach().cpu()))
            self.writer.add_image(f"{part}/melspec_hr_{i}", plot_spectrogram(mel_hr[i].detach().cpu()))
            self.writer.add_image(f"{part}/melspec_pred_{i}", plot_spectrogram(mel_fake[i].detach().cpu()))
