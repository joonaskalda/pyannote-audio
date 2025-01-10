from __future__ import annotations

from typing import Dict, Optional, Sequence, Union, Iterator, List
import math
import os
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchmetrics import Metric
from torch.utils.data.dataloader import default_collate
import pytorch_metric_learning.losses
from pyannote.core import Segment
from pyannote.database import Protocol
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.tasks.embedding.mixins import (
    SupervisedRepresentationLearningTaskMixin,
)


def create_rng_for_worker(model) -> np.random.Generator:
    """Create worker-specific random number generator

    This makes sure that
    1. training samples generation is reproducible
    2. every (worker, epoch) uses a different seed

    Parameters
    ----------
    model : YourModelType
        The model instance containing training state.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator with a unique seed.
    """

    # create random number generator
    global_seed = os.environ.get("PL_GLOBAL_SEED", "unset")
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        worker_id = None
    else:
        worker_id = worker_info.id

    seed_tuple = (
        global_seed,
        worker_id,
        model.local_rank,
        model.global_rank,
        model.current_epoch,
    )
    # use adler32 because python's `hash` is not deterministic.
    seed = zlib.adler32(str(seed_tuple).encode())

    return np.random.default_rng(seed)


class GSE(
    SupervisedRepresentationLearningTaskMixin,
):
    """Implemnentation of https://arxiv.org/pdf/2410.12182

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    cache : str, optional
        As (meta-)data preparation might take a very long time for large datasets,
        it can be cached to disk for later (and faster!) re-use.
        When `cache` does not exist, `Task.prepare_data()` generates training
        and validation metadata from `protocol` and save them to disk.
        When `cache` exists, `Task.prepare_data()` is skipped and (meta)-data
        are loaded from disk. Defaults to a temporary path.
    duration : float, optional
        Chunks duration in seconds. Defaults to two seconds (2.).
    min_duration : float, optional
        Sample training chunks duration uniformely between `min_duration`
        and `duration`. Defaults to `duration` (i.e. fixed length chunks).
    num_classes_per_batch : int, optional
        Number of classes per batch. Defaults to 32.
    num_chunks_per_class : int, optional
        Number of chunks per class. Defaults to 1.
    margin : float, optional
        Margin. Defaults to 28.6.
    scale : float, optional
        Scale. Defaults to 64.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    cache : string, optional
    sampling_mode : str, optional
        Sampling mode. Defaults to "classes_weighted_file_uniform".
        Options are "classes_weighted_file_uniform", "espnet", and "original".
        "classes_weighted_file_uniform" has classes weighted by the total file count.
        "original" has classes weighted by the total duration.
        "espnet" uses exactly (?) the same sampling method as in EspNet-SPK.
    drop_last : bool, optional
        Drop last batch if True. Defaults to False.

    gradient: dict, optional
        Keywords arguments for gradient calculation.
        Defaults to {"clip_val": 5.0, "clip_algorithm": "norm", "accumulate_batches": 1}
    """

    # Class-level constants for magic numbers
    NUM_MIXTURE_SIZE: int = 3
    DEFAULT_SAMPLE_RATE: int = 16000
    MIN_DELAY: float = 0.0
    MAX_DELAY: float = 3.0
    SAMPLE_DURATION_MIN: float = 3.0
    SAMPLE_DURATION_MAX: float = 6.0
    GAIN_DB_RANGE: tuple = (-5, 5)
    ACTIVATION_THRESHOLD: float = 1e-6
    MIN_ACTIVE_DURATION_RATIO: float = 0.05

    def __init__(
        self,
        protocol: Protocol,
        cache: Optional[str] = None,
        min_duration: Optional[float] = None,
        duration: float = 2.0,
        num_classes_per_batch: int = 32,
        num_chunks_per_class: int = 1,
        margin: float = 28.6,
        scale: float = 64.0,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        sampling_mode: str = "original",
        drop_last: bool = False,
        gradient: dict = {
            "clip_val": 5.0,
            "clip_algorithm": "norm",
            "accumulate_batches": 1,
        },
        noise_augmentation: Optional[BaseWaveformTransform] = None,
        rir_augmentation: Optional[BaseWaveformTransform] = None,
    ):
        if num_classes_per_batch % self.NUM_MIXTURE_SIZE != 0:
            raise ValueError(
                f"num_classes_per_batch must be divisible by {self.NUM_MIXTURE_SIZE}"
            )

        self.num_chunks_per_class = num_chunks_per_class
        self.num_classes_per_batch = num_classes_per_batch
        self.margin = margin
        self.scale = scale
        self.drop_last = drop_last
        self.sampling_mode = sampling_mode
        self.gradient = gradient
        self.noise_augmentation = noise_augmentation
        self.rir_augmentation = rir_augmentation

        super().__init__(
            protocol,
            duration=duration,
            min_duration=min_duration,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
            cache=cache,
        )

    def setup_loss_func(self) -> None:
        """Initialize the ArcFace loss function if not already present."""
        if not hasattr(self.model, "arc_face_loss"):
            self.model.arc_face_loss = pytorch_metric_learning.losses.ArcFaceLoss(
                len(self.specifications.classes),
                192,
                margin=self.margin,
                scale=self.scale,
                weight_init_func=nn.init.xavier_normal_,
            )
        self.model.eval()
        _, embedding_size = self.model(self.model.example_input_array).shape
        self.model.train()

    def train__iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator for training batches."""
        rng = create_rng_for_worker(self.model)
        classes = list(self.specifications.classes)

        num_utterances = 0
        num_classes = 0
        X_utterances: List[torch.Tensor] = []
        y_utterances: List[int] = []
        delays: List[float] = []
        utterance_durations: List[float] = []

        while True:
            sampled_classes = self.sample_classes(rng, classes)
            for klass in sampled_classes:
                utterance_duration = rng.uniform(
                    self.SAMPLE_DURATION_MIN, self.SAMPLE_DURATION_MAX
                )
                delay = self.calculate_delay(rng, utterance_durations, delays)
                utterance_durations.append(utterance_duration)
                delays.append(delay)
                y = self.specifications.classes.index(klass)
                file = rng.choice(self.prepared_data["train"][klass])
                speech_turn = self.select_speech_turn(rng, file, utterance_duration)

                X = self.process_speech_turn(
                    file, rng, speech_turn, utterance_duration, delay
                )
                X_utterances.append(X)
                y_utterances.append(y)
                num_utterances += 1
                num_classes += 1

                if len(X_utterances) == self.NUM_MIXTURE_SIZE:
                    # We now have enough utterances to create a mixture
                    mixture = self.create_mixtures(X_utterances, delays, rng=rng)
                    activation_features = self.compute_activation_features(
                        X_utterances, delays
                    )
                    activation_features = activation_features.to(self.model.device)
                    for i in range(self.NUM_MIXTURE_SIZE):
                        yield {
                            "X": mixture,
                            "y": y_utterances[i],
                            "y_active": activation_features[i],
                        }
                    num_utterances = 0
                    X_utterances = []
                    y_utterances = []
                    delays = []
                    utterance_durations = []

                if num_classes == self.batch_size:
                    num_classes = 0

    def sample_classes(self, rng: np.random.Generator, classes: List[str]) -> List[str]:
        """Sample a set of unique classes for the batch based on class probabilities."""
        num_files_per_class = [
            len(self.prepared_data["train"][klass]) for klass in classes
        ]
        total_files = sum(num_files_per_class)
        class_probs = [w / total_files for w in num_files_per_class]

        sampled_classes = list(
            rng.choice(
                classes,
                size=self.num_classes_per_batch,
                replace=False,
                p=class_probs,
            )
        )
        return sampled_classes

    def calculate_delay(
        self,
        rng: np.random.Generator,
        utterance_durations: List[float],
        delays: List[float],
    ) -> float:
        """Calculate the delay for the current utterance."""
        if not utterance_durations:
            # First utterance can be midway by the start of the mixture
            delay = rng.uniform(-self.MAX_DELAY, self.MAX_DELAY)
            return max(delay, self.MIN_DELAY)
        else:
            previous_delay = delays[-1]
            previous_duration = utterance_durations[-1]
            delay = rng.uniform(0.5, previous_duration) + previous_delay
            return delay

    def select_speech_turn(
        self, rng: np.random.Generator, file: Dict, utterance_duration: float
    ) -> Segment:
        """Select a speech turn from the file, weighted by duration."""
        if not file["speech_turns"]:
            raise ValueError("No speech turns available in the selected file.")
        # Compute normalized weights (probabilities)
        durations = [s.duration for s in file["speech_turns"]]
        total_duration = sum(durations)
        probabilities = [d / total_duration for d in durations]

        # Sample one speech turn using probabilities
        speech_turn = rng.choice(
            file["speech_turns"],
            p=probabilities,
        )
        return speech_turn

    def process_speech_turn(
        self,
        file: Dict,
        rng: np.random.Generator,
        speech_turn: Segment,
        utterance_duration: float,
        delay: float,
    ) -> torch.Tensor:
        """Process the speech turn by cropping or padding as necessary."""
        if speech_turn.duration < utterance_duration:
            X, _ = self.model.audio.crop(file, speech_turn)
            num_missing_samples = (
                math.floor(utterance_duration * self.model.audio.sample_rate) - X.shape[1]
            )
            X = F.pad(X, (0, num_missing_samples))
        else:
            start_time = (
                rng.uniform(speech_turn.start, speech_turn.end - utterance_duration)
                if delay == 0.0
                else 0.0
            )
            chunk = Segment(start_time, start_time + utterance_duration)
            X, _ = self.model.audio.crop(
                file,
                chunk,
                duration=utterance_duration,
            )
        return X

    def create_mixtures(
        self,
        X_utterances: List[torch.Tensor],
        delays: List[float],
        rng: np.random.Generator,
        target_len: float = 10.0,
        fs: int = 16000,
    ) -> torch.Tensor:
        """Create a mixture of audio utterances with random delays and gains."""
        target_len_samples = int(target_len * fs)
        mixture = torch.zeros(1, target_len_samples)

        for utt, delay in zip(X_utterances, delays):
            start = int(delay * fs)
            end = start + utt.shape[1]

            if end > target_len_samples:
                end = target_len_samples
                utt = utt[:, : end - start]
            if end - start < target_len_samples * self.MIN_ACTIVE_DURATION_RATIO:
                continue  # Skip if the utterance segment is too short

            gain_db = rng.uniform(*self.GAIN_DB_RANGE)
            gain = self.db_to_amplitude(gain_db)
            mixture[:, start:end] += utt * gain

        if mixture.sum() == 0:
            raise ValueError("Mixture is silent. Adjust delays or durations.")

        return mixture

    @staticmethod
    def db_to_amplitude(db: float) -> float:
        """Convert decibel value to amplitude."""
        return 10 ** (db / 20)

    def compute_activation_features(
        self, X_utterances, delays, target_len=10.0, num_frames=499, fs=16000
    ):
        """
        Compute activation features for each speaker (n_speakers, 2, n_frames)
        First channel: speaker i active frames
        Second channel: any speaker other than i active at frame
        num_frames: lenght of input to ECAPA-TDNN (after signal passed through WavLM)
        """
        target_len_samples = int(target_len * fs)
        # Create binary activity signals for each speaker
        # Shape for each: (1, 1, total_len)
        binary_signals = []
        for utt, delay in zip(X_utterances, delays):
            binary = (utt.abs() > self.ACTIVATION_THRESHOLD).float()
            start = int(delay * fs)
            pad_front = torch.zeros(1, start)
            pad_end_size = target_len_samples - start - binary.shape[1]
            pad_end = torch.zeros(1, max(pad_end_size, 0))
            padded = torch.cat([pad_front, binary, pad_end], dim=1)[
                :, :target_len_samples
            ]
            binary_signals.append(padded.unsqueeze(0))

        binary_tensor = torch.cat(
            binary_signals, dim=0
        )  # Shape: (num_speakers, 1, total_len)

        interp_signals = F.interpolate(
            binary_tensor, size=num_frames, mode="nearest"
        )  # Shape: (num_speakers, 1, num_frames)

        activation_features = torch.zeros(len(X_utterances), 2, num_frames)

        activation_features[:, 0, :] = interp_signals.squeeze(1)

        for i in range(len(X_utterances)):
            others = binary_tensor[torch.arange(len(X_utterances)) != i]
            if others.numel() == 0:
                activation_features[i, 1, :] = 0.0
                continue
            others_max = others.max(dim=0).values
            others_interp = F.interpolate(
                others_max.unsqueeze(0), size=num_frames, mode="nearest"
            )
            activation_features[i, 1, :] = others_interp.squeeze()

        return activation_features

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Perform a training step with the given batch."""
        X, y = batch["X"], batch["y"]
        y_active = batch["y_active"]

        batch_mask = (y_active.sum(-1) > 0)[:, 0]
        X = X[batch_mask]
        y = y[batch_mask]
        y_active = y_active[batch_mask]

        if not self.model.automatic_optimization:
            optimizers = self.model.optimizers()
            optimizers = optimizers if isinstance(optimizers, list) else [optimizers]

            num_accumulate_batches = self.gradient.get("accumulate_batches", 1)
            if batch_idx % num_accumulate_batches == 0:
                for optimizer in optimizers:
                    optimizer.zero_grad()

            loss = self.model.arc_face_loss(self.model(X, y_active), y)
            scaled_loss = loss / num_accumulate_batches
            self.model.manual_backward(scaled_loss)

            if (batch_idx + 1) % num_accumulate_batches == 0:
                for optimizer in optimizers:
                    self.model.clip_gradients(
                        optimizer,
                        gradient_clip_val=self.gradient.get("clip_val", 5.0),
                        gradient_clip_algorithm=self.gradient.get(
                            "clip_algorithm", "norm"
                        ),
                    )
                    optimizer.step()

            if torch.isnan(scaled_loss):
                return None

            self.model.log(
                "loss/train",
                scaled_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            return {"loss": scaled_loss}

        loss = self.model.arc_face_loss(self.model(X, y_active), y)

        if torch.isnan(loss):
            return None

        self.model.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def collate_fn(
        self, batch: List[Dict], stage: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """Collate function to combine batch samples."""
        collated = default_collate(batch)

        if stage == "train":
            sample_rate = getattr(
                self.model.hparams, "sample_rate", self.DEFAULT_SAMPLE_RATE
            )
            self.augmentation.train(mode=True)
            if self.noise_augmentation:
                self.noise_augmentation.train(mode=True)
                collated["X"] = self.noise_augmentation(
                    samples=collated["X"], sample_rate=sample_rate
                )["samples"]
            if self.rir_augmentation:
                self.rir_augmentation.train(mode=True)
                collated["X"] = self.rir_augmentation(
                    samples=collated["X"], sample_rate=sample_rate
                )["samples"]

        return collated
