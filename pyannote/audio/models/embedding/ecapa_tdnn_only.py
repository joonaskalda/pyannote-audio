# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool


class MelSpectrogramEmbeddings(Model):
    """Mel-Spectrogram Representation for Speaker Embeddings Extraction

    Mel-Spectrogram > Stats Pooling > Feed Forward

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    n_mels : int, optional
        Number of mel bins. Defaults to 80.
    n_fft : int, optional
        Number of FFT components. Defaults to 512.
    hop_length : int, optional
        Number of audio samples between adjacent STFT columns. Defaults to 160.
    win_length : int, optional
        Each frame of audio is windowed by `win_length` samples. Defaults to 400.
    emb_dim : int, optional
        Dimension of the speaker embedding in output. Defaults to 192.
    power : float, optional
        Exponent for the magnitude spectrogram. Defaults to 2.0.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        power: float = 2.0,
        emb_dim: Optional[int] = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        # Initialize Mel-Spectrogram transformer with configurable parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=self.power,
        )

        self.pooling = StatsPool()
        self.embedding = nn.Sequential(
            nn.Linear(self.n_mels * 2, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

        # Save all mel-spectrogram related hyperparameters along with emb_dim
        self.save_hyperparameters(
            "n_mels",
            "n_fft",
            "hop_length",
            "win_length",
            "power",
            "emb_dim",
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("MelSpectrogramEmbeddings does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        # Calculation based on STFT parameters
        num_frames = (num_samples - self.n_fft) // self.hop_length + 1
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size in samples.
        """
        receptive_field_size = num_frames * self.hop_length + self.n_fft - self.hop_length
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center in samples.
        """
        center = frame * self.hop_length + self.n_fft // 2
        return center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        embeddings : (batch, emb_dim)
        """
        # Ensure waveforms are mono
        waveforms = waveforms.squeeze(1)  # (batch, sample)

        # Compute Mel-Spectrogram
        mel = self.mel_spectrogram(waveforms)  # (batch, n_mels, time)

        # Optional: Apply logarithmic scaling for better numerical stability
        mel = torch.log1p(mel)  # (batch, n_mels, time)

        # Transpose to (batch, time, n_mels) for pooling
        mel = mel.transpose(1, 2)  # (batch, time, n_mels)

        # Apply statistics pooling (mean and variance)
        mel = self.pooling(mel)  # (batch, n_mels * 2)

        # Pass through embedding layers
        embeddings = self.embedding(mel)  # (batch, emb_dim)

        return embeddings


class MelSpectrogram_ECAPA_TDNN(Model):
    """Mel-Spectrogram Representation for Speaker Embeddings Extraction

    Mel-Spectrogram > ECAPA-TDNN

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    n_mels : int, optional
        Number of mel bins. Defaults to 40.
    n_fft : int, optional
        Number of FFT components. Defaults to 400.
    hop_length : int, optional
        Number of audio samples between adjacent STFT columns. Defaults to 160.
    win_length : int, optional
        Each frame of audio is windowed by `win_length` samples. Defaults to 400.
    power : float, optional
        Exponent for the magnitude spectrogram. Defaults to 2.0.
    emb_dim : int, optional
        Dimension of the speaker embedding in output. Defaults to 192.
    freeze_mel : bool, optional
        Whether to freeze mel-spectrogram parameters. Defaults to False.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        power: float = 2.0,
        emb_dim: Optional[int] = 192,
        freeze_mel: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        # Initialize Mel-Spectrogram transformer with configurable parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=self.power,
        )

        # Optionally freeze mel-spectrogram parameters
        self.freeze_mel = freeze_mel
        if self.freeze_mel:
            for param in self.mel_spectrogram.parameters():
                param.requires_grad = False

        # Initialize ECAPA-TDNN with mel-spectrogram input size
        self.ecapa_tdnn = ECAPA_TDNN(input_size=self.n_mels, lin_neurons=emb_dim)

        # Save all mel-spectrogram related hyperparameters along with emb_dim and freeze_mel
        self.save_hyperparameters(
            "n_mels",
            "n_fft",
            "hop_length",
            "win_length",
            "power",
            "emb_dim",
            "freeze_mel",
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("MelSpectrogram_ECAPA_TDNN does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        # Calculation based on STFT parameters
        num_frames = (num_samples - self.n_fft) // self.hop_length + 1
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size in samples.
        """
        receptive_field_size = num_frames * self.hop_length + self.n_fft - self.hop_length
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center in samples.
        """
        center = frame * self.hop_length + self.n_fft // 2
        return center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        embeddings : (batch, emb_dim)
        """
        # Ensure waveforms are mono
        waveforms = waveforms.squeeze(1)  # (batch, sample)

        # Compute Mel-Spectrogram
        mel = self.mel_spectrogram(waveforms)  # (batch, n_mels, time)

        # Optional: Apply logarithmic scaling for better numerical stability
        mel = torch.log1p(mel)  # (batch, n_mels, time)

        # Transpose to (batch, time, n_mels) for ECAPA-TDNN
        mel = mel.transpose(1, 2)  # (batch, time, n_mels)

        # Pass through ECAPA-TDNN
        embeddings = self.ecapa_tdnn(mel)  # (batch, emb_dim)

        return embeddings.squeeze(1)
