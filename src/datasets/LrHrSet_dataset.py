import os
import json
import torch
from torchaudio.functional import resample
from torchaudio.transforms import Spectrogram


from src.utils.aero_utils_sp import match_signal
from src.datasets.base_dataset import BaseDataset  # Импортируй свой базовый класс отсюда

import math
import torchaudio
from torch.nn import functional as F


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels

        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)


            if sr != self.sample_rate:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                   f"{self.sample_rate}, but got {sr}")
            if out.shape[0] != self.channels:
                raise RuntimeError(f"Expected {file} to have shape of "
                                   f"{self.channels}, but got {out.shape[0]}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out


class LrHrDataset(BaseDataset):
    def __init__(
        self,
        json_dir,
        lr_sr,
        hr_sr,
        stride=None,
        segment=None,
        pad=True,
        with_path=False,
        stft=False,
        win_len=64,
        hop_len=16,
        n_fft=4096,
        complex_as_channels=True,
        upsample=True,
        instance_transforms=None,
        limit=None,
        shuffle_index=False,
    ):
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.stft = stft
        self.with_path = with_path
        self.upsample = upsample
        self.complex_as_channels = complex_as_channels

        if stft:
            self.window_length = int(hr_sr / 1000 * win_len)
            self.hop_length = int(hr_sr / 1000 * hop_len)
            self.spectrogram = Spectrogram(
                n_fft=n_fft,
                win_length=self.window_length,
                hop_length=self.hop_length,
                power=None,
            )

        with open(os.path.join(json_dir, "lr.json"), "r") as f:
            lr_meta = json.load(f)
        with open(os.path.join(json_dir, "hr.json"), "r") as f:
            hr_meta = json.load(f)

        # Строим наборы
        lr_set = Audioset(
            lr_meta,
            sample_rate=lr_sr,
            length=segment * lr_sr if segment else None,
            stride=stride * lr_sr if stride else None,
            pad=pad,
            channels=1,
            with_path=True,
        )
        hr_set = Audioset(
            hr_meta,
            sample_rate=hr_sr,
            length=segment * hr_sr if segment else None,
            stride=stride * hr_sr if stride else None,
            pad=pad,
            channels=1,
            with_path=True,
        )

        assert len(hr_set) == len(lr_set)

        # Генерируем index для BaseDataset
        index = []
        for (lr_sig, lr_path), (hr_sig, hr_path) in zip(lr_set, hr_set):
            index.append({
                "lr_path": lr_path,
                "hr_path": hr_path,
            })

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def load_object(self, path_dict):
        """
        Загрузка пары LR и HR по словарю путей.
        """
        lr_path = path_dict["lr_path"]
        hr_path = path_dict["hr_path"]

        lr_sig, _ = torchaudio.load(lr_path)
        hr_sig, _ = torchaudio.load(hr_path)

        if self.upsample:
            lr_sig = resample(lr_sig, self.lr_sr, self.hr_sr)
            lr_sig = match_signal(lr_sig, hr_sig.shape[-1])

        if self.stft:
            hr_sig = torch.view_as_real(self.spectrogram(hr_sig))
            lr_sig = torch.view_as_real(self.spectrogram(lr_sig))
            if self.complex_as_channels:
                Ch, Fr, T, _ = hr_sig.shape
                hr_sig = hr_sig.reshape(2 * Ch, Fr, T)
                lr_sig = lr_sig.reshape(2 * Ch, Fr, T)

        return {"lr": lr_sig, "hr": hr_sig}
