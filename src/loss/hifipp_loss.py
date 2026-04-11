import torch 
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_gt_output, disc_predicted_output):
        loss = 0
        for gt_output, pred_output in zip(disc_gt_output, disc_predicted_output):
            gt_loss = torch.mean((1 - gt_output) ** 2)
            pred_loss = torch.mean(pred_output ** 2)
            loss += gt_loss + pred_loss
        return loss

        
class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dsc_output):
        loss = 0.0
        for predicted in dsc_output:
            pred_loss = torch.mean((1 - predicted) ** 2)
            loss += pred_loss
        return loss
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial, predicted):
        loss = 0
        for disc_initial_feat, disc_pred_feat in zip(initial, predicted):
            for initial_feat, predicted_feat in zip(disc_initial_feat, disc_pred_feat):
                loss += torch.mean(torch.abs(initial_feat - predicted_feat))
        return loss    


class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial_spec, pred_spec):
        return F.l1_loss(pred_spec, initial_spec)
    
class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft_mag(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        window = self.window.to(device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        mag = spec.abs().clamp_min(1e-7)
        return mag

    def forward(self, pred_wav, gt_wav):
        pred_mag = self._stft_mag(pred_wav)
        gt_mag = self._stft_mag(gt_wav)

        sc_loss = torch.norm(gt_mag - pred_mag, p="fro") / (
            torch.norm(gt_mag, p="fro") + 1e-7
        )
        mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(gt_mag))

        return sc_loss + mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(n_fft=512, hop_length=128, win_length=512),
            STFTLoss(n_fft=1024, hop_length=256, win_length=1024),
            STFTLoss(n_fft=2048, hop_length=512, win_length=2048),
        ])

    def forward(self, pred_wav, gt_wav):
        loss = 0.0
        for loss_fn in self.losses:
            loss = loss + loss_fn(pred_wav, gt_wav)
        return loss
    
class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_mel=45.0, lambda_fm=2.0, lambda_adv=1.0, lambda_stft=5.0):
        super().__init__()
        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm
        self.lambda_adv = lambda_adv
        self.lambda_stft = lambda_stft

        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.melspec_loss = MelSpectrogramLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.stft_loss = MultiResolutionSTFTLoss()

    def generator_forward(
        self,
        mpd_gt_features,
        mpd_pred_features,
        msd_gt_features,
        msd_pred_features,
        mpd_pred_output,
        msd_pred_output,
        gt_melspec,
        pred_melspec,
        gt_wav,
        pred_wav,
    ):
        adv_loss = self.generator_loss(mpd_pred_output) + self.generator_loss(msd_pred_output)

        fm_loss = self.fm_loss(mpd_gt_features, mpd_pred_features) + \
                  self.fm_loss(msd_gt_features, msd_pred_features)

        mel_loss = self.melspec_loss(gt_melspec, pred_melspec)
        stft_loss = self.stft_loss(pred_wav, gt_wav)

        total_loss = (
            self.lambda_adv * adv_loss
            + self.lambda_fm * fm_loss
            + self.lambda_mel * mel_loss
            + self.lambda_stft * stft_loss
        )

        return {
            "loss": total_loss,
            "adv_loss": adv_loss,
            "fm_loss": fm_loss,
            "mel_loss": mel_loss,
            "stft_loss": stft_loss,
        }

    def discriminator_forward(
        self,
        mpd_gt_output,
        mpd_pred_output,
        msd_gt_output,
        msd_pred_output,
    ):
        mpd_loss = self.discriminator_loss(mpd_gt_output, mpd_pred_output)
        msd_loss = self.discriminator_loss(msd_gt_output, msd_pred_output)
        total_loss = mpd_loss + msd_loss

        return {
            "loss": total_loss,
            "mpd_loss": mpd_loss,
            "msd_loss": msd_loss,
        }

    def forward(self, mode: str, **kwargs):
        if mode == "generator":
            return self.generator_forward(**kwargs)
        elif mode == "discriminator":
            return self.discriminator_forward(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")