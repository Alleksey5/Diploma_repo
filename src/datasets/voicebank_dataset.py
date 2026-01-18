import os
import random
import torch
import librosa
from src.datasets.base_dataset import BaseDataset


class VoicebankDataset(BaseDataset):
    def __init__(
        self,
        noisy_wavs_dir,
        clean_wavs_dir=None,
        path_prefix=None,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
    ):
        """
        Класс для работы с Voicebank датасетом.
        Наследует базовую логику от BaseDataset.
        """

        if path_prefix:
            if clean_wavs_dir:
                clean_wavs_dir = os.path.join(path_prefix, clean_wavs_dir)
            noisy_wavs_dir = os.path.join(path_prefix, noisy_wavs_dir)
        
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        audio_files = self.read_files_list(clean_wavs_dir, noisy_wavs_dir)
        
        random.seed(1234)
        if shuffle:
            random.shuffle(audio_files)

        super().__init__(data_paths=audio_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq

    @staticmethod
    def read_files_list(clean_wavs_dir, noisy_wavs_dir):
        """
        Читает список файлов из папок чистого и шумного аудио.
        """
        fn_lst_clean = os.listdir(clean_wavs_dir)
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        assert set(fn_lst_clean) == set(fn_lst_noisy), "Списки файлов не совпадают!"
        return sorted(fn_lst_clean)

    def __getitem__(self, index):
        """
        Загружает и возвращает данные для одного примера.
        """
        fn = self.data_paths[index]

        clean_audio = librosa.load(
            os.path.join(self.clean_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]
        noisy_audio = librosa.load(
            os.path.join(self.noisy_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]

        clean_audio, noisy_audio = split_audios(
            [clean_audio, noisy_audio],
            self.segment_size, self.split
        )

        input_audio = normalize(noisy_audio)[None] * 0.95
        assert input_audio.shape[1] == clean_audio.size, "Несоответствие размеров аудио!"

        input_audio = torch.FloatTensor(input_audio)
        audio = torch.FloatTensor(normalize(clean_audio) * 0.95).unsqueeze(0)

        return input_audio, audio
