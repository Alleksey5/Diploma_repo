from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.metrics.calculate_metrics import calculate_all_metrics
from src.model.hifigan import mel_spectrogram



class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler,
        disc_lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=gen_optimizer,
            lr_scheduler=gen_lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

    def create_mel_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, 1, T] или [B, T]
        return: [B, 80, frames]
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)  # [B, T]

        mel = mel_spectrogram(
            wav,
            n_fft=1024,
            num_mels=80,
            sampling_rate=16000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
            center=False,
        )
        
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [1, 80, T]

        return mel

    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        initial_wav = batch["wav"]     # NB
        gt_wav = batch["gt_wav"]       # WB
        wav_fake = self.model.generator(initial_wav)
 
        # wav_fake: [B, 1, T_fake], gt_wav: [B, 1, T_gt]
        T_fake = wav_fake.shape[-1]
        T_gt = gt_wav.shape[-1]

        if T_fake < T_gt:
            wav_fake = F.pad(wav_fake, (0, T_gt - T_fake))
        elif T_fake > T_gt:
            wav_fake = wav_fake[..., :T_gt]

        batch["generated_wav"] = wav_fake
        mel_spec_fake = self.create_mel_spec(wav_fake)
        batch['mel_spec_fake'] = mel_spec_fake
        if self.is_train:
            self.disc_optimizer.zero_grad()

        mpd_gt_out, _, mpd_fake_out, _ = self.model.mpd(gt_wav, wav_fake.detach())

        msd_gt_out, _,  msd_fake_out, _ = self.model.msd(gt_wav, wav_fake.detach())

        mpd_disc_loss = self.criterion.discriminator_loss(mpd_gt_out, mpd_fake_out)
        msd_disc_loss = self.criterion.discriminator_loss(msd_gt_out, msd_fake_out)
        disc_loss = mpd_disc_loss + msd_disc_loss


        if self.is_train:
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)

        if self.is_train:
            disc_loss.backward()
            self.disc_optimizer.step()
            self.gen_optimizer.zero_grad()




        _, mpd_gt_feats, mpd_fake_out, mpd_fake_feats = self.model.mpd(gt_wav, wav_fake)

        _, msd_gt_features, msd_fake_out, msd_fake_feats = self.model.msd(gt_wav, wav_fake)     

        mpd_gen_loss = self.criterion.generator_loss(mpd_fake_out)
        msd_gen_loss = self.criterion.generator_loss(msd_fake_out)

        # initial_melspec = mel_spectrogram(gt_wav.squeeze(1), 1024, 80, 16000, 256, 1024, 0, 8000)
        melspec_gt = batch['gt_melspec']

        mel_spec_loss = self.criterion.melspec_loss(melspec_gt, mel_spec_fake)
        
        mpd_feats_gen_loss = self.criterion.fm_loss(mpd_gt_feats, mpd_fake_feats)
        msd_feats_gen_loss = self.criterion.fm_loss(msd_gt_features, msd_fake_feats)

        gen_loss = mpd_gen_loss + msd_gen_loss + mel_spec_loss + mpd_feats_gen_loss + msd_feats_gen_loss


        if self.is_train:
            self._clip_grad_norm(self.model.generator)
            gen_loss.backward()
            self.gen_optimizer.step()


        batch["mpd_disc_loss"] = mpd_disc_loss
        batch["msd_disc_loss"] = msd_disc_loss
        batch["disc_loss"] = disc_loss
        batch["mpd_gen_loss"] = mpd_gen_loss
        batch["msd_gen_loss"] = msd_gen_loss
        batch["mel_spec_loss"] = mel_spec_loss
        batch["mpd_feats_gen_loss"] = mpd_feats_gen_loss
        batch["msd_feats_gen_loss"] = msd_feats_gen_loss
        batch["gen_loss"] = gen_loss
        batch["loss"] = gen_loss + disc_loss
    


        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        scores = {}

        if not self.is_train:
            scores = calculate_all_metrics(
                batch["generated_wav"],
                batch["gt_wav"],
                self.metrics["inference"],
                self.config.datasets.val.low_sampling_rate,
                self.config.datasets.val.high_sampling_rate,
            )
            for name, (mean, std) in scores.items():
                metrics.update(name, float(mean))

        for name, (mean, std) in scores.items():
            batch[name] = torch.tensor(mean, device=self.device)


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
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(partition='train', idx=0, **batch)
            self.log_audio(partition='train', idx=0, **batch)

        else:
            # Log Stuff
            self.log_spectrogram(partition='val', idx=batch_idx, **batch)
            self.log_audio(partition='val', idx=batch_idx,**batch)


    def log_audio(self, gt_wav, generated_wav, partition, idx, **batch):
        init_gt_len = batch['initial_gt_len'][0]
        init_len = batch['initial_len'][0]
        if partition != 'val':
            self.writer.add_audio("initial_wav", gt_wav[0][:, :init_gt_len], self.config.datasets.train.high_sampling_rate)
            self.writer.add_audio("generated_wav", generated_wav[0][:, :init_len], self.config.datasets.train.high_sampling_rate)
        else:
            self.writer.add_audio(f"initial_wav_{idx}", gt_wav[0][:, :init_gt_len], self.config.datasets.val.high_sampling_rate)
            self.writer.add_audio(f"generated_wav_{idx}", generated_wav[0][:, :init_len], self.config.datasets.val.high_sampling_rate)


    def log_spectrogram(self,gt_melspec,  mel_spec_fake, partition, idx, **batch):

        spectrogram_for_plot_real = gt_melspec[0].detach().cpu()[:, :batch['initial_gt_melspec_len'][0]]
        spectrogram_for_plot_fake = mel_spec_fake[0].detach().cpu()[:, :batch['initial_melspec_len'][0]]
        image = plot_spectrogram(spectrogram_for_plot_real)
        self.writer.add_image("melspectrogram_real", image)
        image_fake = plot_spectrogram(spectrogram_for_plot_fake)
        self.writer.add_image("melspectrogram_fake", image_fake)

