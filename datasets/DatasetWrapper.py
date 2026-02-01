# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Optional
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
from abc import ABCMeta
from random import Random
from numpy import repeat
from itertools import chain


class DatasetWrapper(Dataset[T_co], metaclass=ABCMeta):
    basic_transform: Callable[[T_co], T_co]
    augment_transform: Callable[[T_co], T_co]

    def __init__(
        self,
        labels: Iterable[int],
        base_ratio: float,
        num_phases: int,
        augment: bool,
        inplace_repeat: int = 1,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        # Type hints
        self.dataset: Dataset[T_co]
        self.num_classes: int

        # Initialization
        super().__init__()
        self.inplace_repeat = inplace_repeat
        self.base_ratio = base_ratio
        self.num_phases = num_phases
        self.base_size = int(self.num_classes * self.base_ratio)
        self.incremental_size = self.num_classes - self.base_size
        self.phase_size = self.incremental_size // num_phases if num_phases > 0 else 0
        # Create a list of indices for each class
        self.class_indices: list[list[int]] = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        self._transform = self.augment_transform if augment else self.basic_transform
        # Shuffle the class indices
        self.real_labels: list[int] = list(range(self.num_classes))
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(self.real_labels)
            Random(shuffle_seed).shuffle(self.class_indices)

    def __getitem__(self, index: int) -> T_co:
        return self._transform(self.dataset[index])

    def _subset(self, label_begin: int, label_end: int) -> Subset[T_co]:
        sub_ids = tuple(chain.from_iterable(self.class_indices[label_begin:label_end]))
        return Subset(self, repeat(sub_ids, self.inplace_repeat).tolist())

    def subset_at_phase(self, phase: int) -> Subset[T_co]:
        if phase == 0:
            return self._subset(0, self.base_size)
        return self._subset(
            self.base_size + (phase - 1) * self.phase_size,
            self.base_size + phase * self.phase_size,
        )

    def subset_until_phase(self, phase: int) -> Subset[T_co]:
        return self._subset(
            0,
            self.base_size + phase * self.phase_size,
        )

import cv2
import numpy as np
import torch
from scipy.io import wavfile
import librosa
from torchaudio.transforms import MelSpectrogram
import math
from torchaudio.functional import amplitude_to_DB
from PIL import Image


audio_opts = {
    'sample_rate': 16000,
    'n_fft': 512,
    'win_length': 320,
    'hop_length': 160,
    'n_mel': 80,
}

def load_wav(path, fr=0, to=10000, sample_rate=16000):
    """Loads Audio wav from path at time indices given by fr, to (seconds)"""

    _, wav = wavfile.read(path)
    '''
    fr_aud = int(np.round(fr * sample_rate))
    to_aud = int(np.round((to) * sample_rate))

    wav = wav[fr_aud:to_aud]
    '''
    return wav

def wav2filterbanks(wav, mel_basis=None, resnet=False, device="cpu"):
    """
    :param wav: Tensor b x T
    """

    assert len(wav.shape) == 2, 'Need batch of wavs as input'

    spect = torch.stft(wav,
                       n_fft=audio_opts['n_fft'],
                       hop_length=audio_opts['hop_length'],
                       win_length=audio_opts['win_length'],
                       window=torch.hann_window(audio_opts['win_length']).to(device),
                       center=True,
                       pad_mode='reflect',
                       normalized=False,
                       onesided=True,
                       return_complex=False)  # b x F x T x 2
    spect = spect[:, :, :-1, :]

    # ----- Log filterbanks --------------
    # mag spectrogram - # b x F x T
    mag = power_spect = torch.norm(spect, dim=-1)
    phase = torch.atan2(spect[..., 1], spect[..., 0])
    if resnet:
        features = mag.permute([0, 2, 1])
        return features
    if mel_basis is None:
        # Build a Mel filter
        mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=audio_opts['sample_rate'],
                                n_fft=audio_opts['n_fft'],
                                n_mels=audio_opts['n_mel'],
                                fmin=0,
                                fmax=int(audio_opts['sample_rate'] / 2)))
        mel_basis = mel_basis.float().to(power_spect.device)
    features = torch.log(torch.matmul(mel_basis, power_spect) + 1e-20)  # b x F x T
    features = features.permute([0, 2, 1]).contiguous()  # b x T x F
    # -------------------

    # norm_axis = 1 # normalize every sample over time
    # mean = features.mean(dim=norm_axis, keepdim=True) # b x 1 x F
    # std_dev = features.std(dim=norm_axis, keepdim=True) # b x 1 x F
    # features = (features - mean) / std_dev # b x T x F

    return features, mag, phase, mel_basis

def wave2input(wav, device):
    transform = MelSpectrogram(sample_rate=audio_opts['sample_rate'],
     win_length=audio_opts['win_length'], hop_length=audio_opts['hop_length'], 
     n_fft=audio_opts['n_fft'], n_mels=audio_opts['n_mel']).to(device)
    mel_wav = transform(wav)
    wav_2_db = amplitude_to_DB(mel_wav, multiplier=10, amin=1e-10, db_multiplier=math.log10(max(1e-10, torch.max(mel_wav).item())), top_db=80)
    wav_2_db = (wav_2_db + 40) / 40
    return wav_2_db.permute([0, 2, 1])


class DatasetWrapper_DIL(Dataset[T_co], metaclass=ABCMeta):
    basic_transform: Callable[[T_co], T_co]
    augment_transform: Callable[[T_co], T_co]

    def __init__(
        self,
        labels: Iterable[int],
        base_ratio: float,
        num_phases: int,
        augment: bool,
        inplace_repeat: int = 1,
        shuffle_seed: Optional[int] = None,
        waveform_form = False,
        fine_grained_labels = False,
    ) -> None:
        # Type hints
        self.dataset: Dataset[T_co]
        self.num_domains: int
        self.domain_indices: list[list[int]]

        # Initialization
        super().__init__()
        self.inplace_repeat = inplace_repeat
        self.base_ratio = base_ratio
        self.num_phases = num_phases
        self.base_size = max(int(self.num_domains * self.base_ratio), 2)
        self.incremental_size = self.num_domains - self.base_size
        self.phase_size = self.incremental_size // num_phases if num_phases > 0 else 0
        self._transform = self.augment_transform if augment else self.basic_transform
        self.waveform_form = waveform_form
        self.fine_grained_labels = fine_grained_labels
        # Shuffle the domain indices
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(self.domain_indices)

    def __getitem__(self, index: int) -> T_co:
        fpath_list, audio, label, duration = self.dataset[index]
        # Total number of sampled frames.
        len_list = len(fpath_list)
        frame_N = len_list

        # read video frames and resize to a fixed size
        buffer = torch.empty((3, self.seq_len, self.resize, self.resize), dtype=torch.float)
        idx = 0
        for idx, i in enumerate(range(frame_N)):
            fpath = fpath_list[i]
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = self._transform(image=img)['image']
            img = Image.fromarray(img)
            img = self._transform(img)
            buffer[:,idx, ...] = img

        
        # read audio and convert to filterbanks
        fps = 25  # TODO: get as param?
        aud_fact = int(np.round(audio_opts['sample_rate'] / fps))  #一帧有多少数据点
        audio = load_wav(audio).astype('float32')
        audio = audio[:frame_N * aud_fact]  # truncate to match video length
        audio = torch.from_numpy(audio).float()
        if not self.waveform_form:
            # transform audio to mel spectrogram
            audio, _, _, _ = wav2filterbanks(audio.unsqueeze(0))
            audio = audio.squeeze(0)
            audio = audio.permute([1, 0])  # b x T x F

        if self.fine_grained_labels:
            if label == 0:
                video_label = 0
                audio_label = 0
                total_label = 0
            elif label == 1:
                video_label = 1
                audio_label = 1
                total_label = 1
            elif label == 2 :
                video_label = 1
                audio_label = 0
                total_label = 2
            else:
                video_label = 0
                audio_label = 1
                total_label = 3
            video_label = torch.tensor(video_label, dtype=torch.long)
            audio_label = torch.tensor(audio_label, dtype=torch.long)
            total_label = torch.tensor(total_label, dtype=torch.long)

            buffer = buffer.view(-1, self.resize, self.resize)
            return (buffer, audio), (video_label, audio_label, total_label)
        else:
            # Binary classification label
            label = torch.tensor(int(label), dtype=torch.long)
            
            # return {'img': buffer, 'audio': audio, 'label': label}
            return (buffer, audio), label

    def _subset(self, label_begin: int, label_end: int) -> Subset[T_co]:
        sub_ids = tuple(chain.from_iterable(self.domain_indices[label_begin:label_end]))
        return Subset(self, repeat(sub_ids, self.inplace_repeat).tolist())

    def subset_at_phase(self, phase: int) -> Subset[T_co]:
        if phase == 0:
            return self._subset(0, self.base_size)
        return self._subset(
            self.base_size + (phase - 1) * self.phase_size,
            self.base_size + phase * self.phase_size,
        )

    def subset_until_phase(self, phase: int) -> Subset[T_co]:
        return self._subset(
            0,
            self.base_size + phase * self.phase_size,
        )