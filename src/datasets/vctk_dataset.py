import logging
import os
import random

import librosa
import numpy as np
import scipy.signal
import torch
from librosa.util import normalize

from src.datasets.base_dataset import BaseDataset
from src.model.melspec import MelSpectrogram, MelSpectrogramConfig


logger = logging.getLogger(__name__)


def get_dataset_filelist(dataset_split_file, input_wavs_dir):
    """
    Reads relative paths from split file and joins them with root dir.
    Missing files are skipped with warnings.
    """
    existing_files = []
    missing_files = []

    with open(dataset_split_file, "r", encoding="utf-8") as fi:
        for raw_fn in fi.read().splitlines():
            fn = raw_fn.strip()
            if not fn:
                continue

            full_path = os.path.join(input_wavs_dir, fn)
            if os.path.isfile(full_path):
                existing_files.append(full_path)
            else:
                missing_files.append(full_path)

    if missing_files:
        logger.warning(
            f"Found {len(missing_files)} missing audio files in split "
            f"'{dataset_split_file}'. They will be skipped."
        )
        for path in missing_files[:10]:
            logger.warning(f"Missing audio skipped: {path}")
        if len(missing_files) > 10:
            logger.warning(f"... and {len(missing_files) - 10} more missing files.")

    if not existing_files:
        raise RuntimeError(
            f"No valid audio files found after filtering. "
            f"split='{dataset_split_file}', root='{input_wavs_dir}'"
        )

    return existing_files


def lowpass_and_downsample(
    audio: np.ndarray,
    high_sampling_rate: int,
    low_sampling_rate: int,
    lp_type: str = "default",
) -> np.ndarray:
    """
    HiFi++-style degradation for BWE:
      1) low-pass with cutoff = low_sampling_rate / 2
      2) downsample from high_sampling_rate to low_sampling_rate

    For 4k -> 16k task:
      high_sampling_rate = 16000
      low_sampling_rate  = 4000
      cutoff             = 2000 Hz
    """
    if low_sampling_rate is None:
        raise ValueError("low_sampling_rate must be provided for BWE task.")

    if low_sampling_rate >= high_sampling_rate:
        raise ValueError(
            f"low_sampling_rate ({low_sampling_rate}) must be smaller than "
            f"high_sampling_rate ({high_sampling_rate})"
        )

    if lp_type == "default":
        low_audio = librosa.resample(
            audio,
            orig_sr=high_sampling_rate,
            target_sr=low_sampling_rate,
            res_type="polyphase",
        )

    elif lp_type == "decimate":
        ratio = high_sampling_rate / low_sampling_rate
        if int(ratio) != ratio:
            raise ValueError(
                f"For decimate mode, high_sampling_rate ({high_sampling_rate}) "
                f"must be divisible by low_sampling_rate ({low_sampling_rate})."
            )
        low_audio = scipy.signal.decimate(audio, int(ratio), ftype="iir", zero_phase=True)

    else:
        raise NotImplementedError(f"Unknown lowpass mode: {lp_type}")

    return low_audio.astype(np.float32)


def upsample_to_high_rate(
    audio: np.ndarray,
    low_sampling_rate: int,
    high_sampling_rate: int,
) -> np.ndarray:
    """
    Upsamples low-rate waveform back to high-rate grid.

    Important:
    after this operation the waveform has sample rate = high_sampling_rate,
    but its information content is still limited by the low-rate source.
    """
    up_audio = librosa.resample(
        audio,
        orig_sr=low_sampling_rate,
        target_sr=high_sampling_rate,
        res_type="polyphase",
    )
    return up_audio.astype(np.float32)


def split_audios(audios, segment_size, split):
    """
    Splits or pads all audios to the same segment_size measured in samples.
    Expects all audios already on the same sampling-rate grid.
    """
    audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]

    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start: audio_start + segment_size]
                for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]

    audios = [audio.squeeze(0).numpy() for audio in audios]
    return audios


class VCTKDataset(BaseDataset):
    """
    BWE dataset for 4k -> 16k task.

    Returned tensors:
      wav        : degraded input on 16k grid (created as 16k -> 4k -> 16k)
      gt_wav     : target wideband waveform at 16k
      melspec    : mel of degraded input
      gt_melspec : mel of target waveform
    """

    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir,
        segment_size=32768,
        high_sampling_rate=16000,
        low_sampling_rate=4000,
        split=True,
        shuffle=False,
        device=None,
        lowpass="default",
    ):
        self.audio_files = get_dataset_filelist(dataset_split_file, vctk_wavs_dir)

        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

        self.segment_size = int(segment_size)
        self.high_sampling_rate = int(high_sampling_rate)
        self.low_sampling_rate = int(low_sampling_rate) if low_sampling_rate is not None else None
        self.split = split
        self.device = device
        self.lowpass = lowpass
        self.clean_wavs_dir = vctk_wavs_dir

        if self.low_sampling_rate is None:
            raise ValueError("low_sampling_rate must be specified for BWE.")

        if self.low_sampling_rate >= self.high_sampling_rate:
            raise ValueError(
                f"Expected low_sampling_rate < high_sampling_rate, got "
                f"{self.low_sampling_rate} >= {self.high_sampling_rate}"
            )

        self.mel_creator = MelSpectrogram(
            MelSpectrogramConfig(sr=self.high_sampling_rate)
        )

    def __getitem__(self, index):
        if len(self.audio_files) == 0:
            raise IndexError("Dataset is empty.")

        start_index = index

        while True:
            vctk_fn = self.audio_files[index]
            try:
                # 1) Load target waveform in 16 kHz.
                gt_audio = librosa.load(
                    vctk_fn,
                    sr=self.high_sampling_rate,
                    res_type="polyphase",
                )[0].astype(np.float32)
                break
            except Exception as e:
                logger.warning(f"Failed to load '{vctk_fn}'. Skipping. Error: {e}")
                index = (index + 1) % len(self.audio_files)
                if index == start_index:
                    raise RuntimeError("Failed to load any audio file from dataset.")

        (gt_audio,) = split_audios([gt_audio], self.segment_size, self.split)

        low_audio = lowpass_and_downsample(
            gt_audio,
            high_sampling_rate=self.high_sampling_rate,
            low_sampling_rate=self.low_sampling_rate,
            lp_type=self.lowpass,
        )

        input_audio = upsample_to_high_rate(
            low_audio,
            low_sampling_rate=self.low_sampling_rate,
            high_sampling_rate=self.high_sampling_rate,
        )

        if input_audio.shape[0] < gt_audio.shape[0]:
            input_audio = np.pad(
                input_audio,
                (0, gt_audio.shape[0] - input_audio.shape[0]),
                mode="constant",
            )
        elif input_audio.shape[0] > gt_audio.shape[0]:
            input_audio = input_audio[: gt_audio.shape[0]]

        input_audio = normalize(input_audio)[None] * 0.95
        gt_audio = normalize(gt_audio)[None] * 0.95

        assert input_audio.shape[1] == gt_audio.shape[1], (
            f"Shape mismatch after resampling: "
            f"input={input_audio.shape}, gt={gt_audio.shape}"
        )

        input_audio = torch.FloatTensor(input_audio)
        gt_audio = torch.FloatTensor(gt_audio)

        melspec = self.mel_creator(input_audio.detach()).squeeze(0)
        gt_melspec = self.mel_creator(gt_audio.detach()).squeeze(0)

        return {
            "wav": input_audio,
            "gt_wav": gt_audio,
            "path": vctk_fn,
            "melspec": melspec,
            "gt_melspec": gt_melspec,
        }

    def __len__(self):
        return len(self.audio_files)