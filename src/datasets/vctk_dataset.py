import os
import random
import torch
import librosa
import numpy as np
from librosa.util import normalize
from scipy import signal
from src.datasets.base_dataset import BaseDataset


class VCTKDataset(BaseDataset):
    """
    VCTK Dataset implementation inheriting from BaseDataset.
    This dataset handles VCTK audio files for speech processing tasks.
    """

    def __init__(
        self, dataset_split_file, vctk_wavs_dir, segment_size=8192, sampling_rate=16000, input_freq = 1000, window=4096,
        split=True, shuffle=False, instance_transforms=None, limit=None
    ):
        """
        Args:
            dataset_split_file (str): Path to the dataset split file.
            vctk_wavs_dir (str): Directory where VCTK WAV files are stored.
            segment_size (int): Size of each audio segment.
            sampling_rate (int): Sampling rate for audio.
            split (bool): Whether to split the audio into segments.
            shuffle (bool): Whether to shuffle the dataset.
            instance_transforms (dict | None): Dictionary of transforms to apply to instances.
            limit (int | None): Maximum number of dataset elements to include.
        """
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.input_freq = input_freq
        self.split = split
        self.window = window

        self.audio_files = self._load_file_list(dataset_split_file, vctk_wavs_dir)
        if shuffle:
            random.shuffle(self.audio_files)

        index = self._create_index()
        super().__init__(index, limit=limit, shuffle_index=False, instance_transforms=instance_transforms)

    def _load_file_list(self, dataset_split_file, vctk_wavs_dir):
        """Load list of audio file paths from dataset split file."""
        with open(dataset_split_file, "r", encoding="utf-8") as f:
            files = [os.path.join(vctk_wavs_dir, line.strip()) for line in f if line.strip()]
        return files

    def _create_index(self):
        """Create index for dataset."""
        return [{"path": file_path} for file_path in self.audio_files]

    def load_object(self, path):
        """Load an audio file from disk and preprocess it."""
        audio, _ = librosa.load(path, sr=self.sampling_rate, res_type="polyphase")
        audio = self.split_audios([audio], self.segment_size, self.split)[0]
        return torch.FloatTensor(normalize(audio) * 0.95)

    def split_audios(self, audios, segment_size, split=True):
        """
        Correctly splits audios into multiple segments of given size.

        Args:
            audios (list[np.ndarray] or list[torch.Tensor]): List of audio signals.
            segment_size (int): Target segment size in samples.
            split (bool): Whether to split the audio into multiple smaller segments.

        Returns:
            list[np.ndarray]: List of processed audio segments.
        """
        window = self.window

        processed_audios = []
        sizes = []

        for audio in audios:
            if isinstance(audio, np.ndarray):
                audio = torch.FloatTensor(audio).unsqueeze(0)

            audio_len = audio.size(1)

            if split:
                start = 0
                while start + segment_size <= audio_len:
                    print(start, start + segment_size)
                    segment = audio[:, start:start + segment_size]
                    processed_audios.append(segment.squeeze(0).numpy())
                    start += window

                par = start + segment_size - window

                if start < audio_len:
                    end_segment = audio[:, -segment_size:]
                    processed_audios.append(end_segment.squeeze(0).numpy())
                    sizes.append(audio_len - par)
                    print(audio_len - par, sizes[0], audio_len)
            else:
                if audio_len < segment_size:
                    pad_size = segment_size - audio_len
                    audio = torch.nn.functional.pad(audio, (0, pad_size), mode="constant", value=0)
                processed_audios.append(audio.squeeze(0).numpy())

        print(f"Total segments created: {len(processed_audios)}")
        return processed_audios, sizes


    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the dataset item.

        Returns:
            dict: Dictionary containing:
                - 'input_audio': Processed low-pass filtered audio.
                - 'audio': Original normalized audio.
        """
        vctk_fn = self.audio_files[index]
        vctk_audio, _ = librosa.load(vctk_fn, sr=self.sampling_rate, res_type="polyphase")

        audio_segments, sizes = self.split_audios([vctk_audio], self.segment_size, self.split)

        batch = []
        for segment in audio_segments:
            lp_inp = self.low_pass_filter(
                                      segment, self.input_freq,
                                      lp_type="default", orig_sr=self.sampling_rate
                                    )

            input_audio = torch.FloatTensor(normalize(lp_inp)[None] * 0.95)  # (1, N)
            audio = torch.FloatTensor(normalize(segment) * 0.95).unsqueeze(0)  # (1, N)

            batch.append({"audio": input_audio, "tg_audio": audio, "file_id": index, "size": sizes[0]})

        return batch

    def __len__(self):
        return len(self.audio_files)

    @staticmethod
    def low_pass_filter(audio, max_freq, orig_sr=16000, lp_type="default"):
        """
        Apply a low-pass filter to the input audio.

        Args:
            audio (np.ndarray): Input audio.
            max_freq (int): Max frequency cutoff.
            orig_sr (int): Original sample rate.
            lp_type (str): Type of low-pass filter.

        Returns:
            np.ndarray: Filtered audio.
        """
        if lp_type == "default":
            tmp = librosa.resample(audio, orig_sr=orig_sr, target_sr=max_freq * 2, res_type="polyphase")
        elif lp_type == "decimate":
            sub = orig_sr / (max_freq * 2)
            if not sub.is_integer():
                raise ValueError("Decimation factor must be an integer.")
            tmp = signal.decimate(audio, int(sub))
        else:
            raise NotImplementedError("Unknown low-pass filter type.")

        return librosa.resample(tmp, orig_sr=max_freq * 2, target_sr=orig_sr, res_type="polyphase")[: len(audio)]
