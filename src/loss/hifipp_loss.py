import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.hifi_utils import mel_spectrogram

class HiFiPPPretrainLoss(nn.Module):
    """
    Pretrain loss for HiFi++ without GAN:
    L = L1(waveform) + mel_weight * L1(mel)
    """
    def __init__(self, mel_weight: float = 45.0):
        super().__init__()
        self.mel_weight = mel_weight

    def forward(self, pred_audio, target_audio, **batch):
        # pred_audio, target_audio: (B, 1, T)
        wav_l1 = F.l1_loss(pred_audio, target_audio)

        # mel_spectrogram expects (B, T) tensor
        pred = pred_audio.squeeze(1)
        tgt = target_audio.squeeze(1)

        pred_mel = mel_spectrogram(pred, 1024, 80, 16000, 256, 1024, 0, 8000)
        tgt_mel  = mel_spectrogram(tgt,  1024, 80, 16000, 256, 1024, 0, 8000)

        mel_l1 = F.l1_loss(pred_mel, tgt_mel)

        loss = wav_l1 + self.mel_weight * mel_l1
        return {"loss": loss, "wav_l1": wav_l1.detach(), "mel_l1": mel_l1.detach()}
