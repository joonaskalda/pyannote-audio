# MIT License
#
# Copyright (c) 2020- CNRS
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

import math
import pickle
from pathlib import Path
from tempfile import mkstemp
from typing import Dict, Sequence, Union

import torch
import torch.nn.functional as F
from pyannote.core import Segment
from pyannote.database.protocol import (
    SpeakerDiarizationProtocol,
    SpeakerVerificationProtocol,
)
from torch.utils.data._utils.collate import default_collate
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.torchmetrics.classification import EqualErrorRate
from pyannote.audio.utils.random import create_rng_for_worker

from enum import Enum
import numpy as np

class SamplingMode(Enum):
    RANDOM_CLASS_WEIGHTED_FILE_DURATION = 1
    CLASS_WEIGHTED_FILE_UNIFORM = 2
    FILE_WEIGHTED_NO_REPLACEMENT = 3

class SupervisedRepresentationLearningTaskMixin(Task):
    # def __init__(self, sampling_mode=SamplingMode.RANDOM_CLASS_WEIGHTED_FILE_DURATION, **kwargs):
    #     super().__init__(**kwargs)
    #     self.sampling_mode = sampling_mode
    #     # Initialize any additional attributes needed for sampling
    #     self.reset_epoch_state()

    """Methods common to most supervised representation tasks"""

    # batch_size = num_classes_per_batch x num_chunks_per_class

    @property
    def num_classes_per_batch(self) -> int:
        if hasattr(self, "num_classes_per_batch_"):
            return self.num_classes_per_batch_
        return self.batch_size // self.num_chunks_per_class

    @num_classes_per_batch.setter
    def num_classes_per_batch(self, num_classes_per_batch: int):
        self.num_classes_per_batch_ = num_classes_per_batch

    @property
    def num_chunks_per_class(self) -> int:
        if hasattr(self, "num_chunks_per_class_"):
            return self.num_chunks_per_class_
        return self.batch_size // self.num_classes_per_batch

    @num_chunks_per_class.setter
    def num_chunks_per_class(self, num_chunks_per_class: int):
        self.num_chunks_per_class_ = num_chunks_per_class

    @property
    def batch_size(self) -> int:
        if hasattr(self, "batch_size_"):
            return self.batch_size_
        return self.num_chunks_per_class * self.num_classes_per_batch

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.batch_size_ = batch_size

    def prepare_data(self):
        # loop over the training set, remove annotated regions shorter than
        # chunk duration, and keep track of the reference annotations, per class.
        if self.cache:
            # check if cache exists and is not empty:
            if self.cache.exists() and self.cache.stat().st_size > 0:
                # data was already created, nothing to do
                return
            # create parent directory if needed
            self.cache.parent.mkdir(parents=True, exist_ok=True)
        else:
            # if no cache was provided by user, create a temporary file
            # in system directory used for temp files
            self.cache = Path(mkstemp()[1])

        train = {}

        desc = f"Loading {self.protocol.name} training labels"
        for f in tqdm(iterable=self.protocol.train(), desc=desc, unit="file"):
            for klass in f["annotation"].labels():
                # keep class's (long enough) speech turns...
                speech_turns = [
                    segment
                    for segment in f["annotation"].label_timeline(klass)
                    if segment.duration > self.min_duration
                ]

                # skip if there is no speech turns left
                if not speech_turns:
                    continue

                # ... and their total duration
                duration = sum(segment.duration for segment in speech_turns)

                # add class to the list of classes
                if klass not in train:
                    train[klass] = list()

                train[klass].append(
                    {
                        "uri": f["uri"],
                        "audio": f["audio"],
                        "duration": duration,
                        "speech_turns": speech_turns,
                    }
                )

        prepared_data = {"train": train, "protocol": self.protocol.name}

        self.prepare_validation(prepared_data)
        self.post_prepare_data(prepared_data)

        # save prepared data on the disk
        with open(self.cache, "wb") as cache_file:
            pickle.dump(prepared_data, cache_file)

    def setup(self, stage=None):
        if stage == "fit":
            self.cache = self.trainer.strategy.broadcast(self.cache)

        try:
            with open(self.cache, "rb") as cache_file:
                self.prepared_data = pickle.load(cache_file)
        except FileNotFoundError:
            print(
                "Cached data for protocol not found. Ensure that prepare_data() was called",
                " and executed correctly or/and that the path to the task cache is correct.",
            )
            raise

        # checks that the task current protocol matches the cached protocol
        if self.protocol.name != self.prepared_data["protocol"]:
            raise ValueError(
                f"Protocol specified for the task ({self.protocol.name}) "
                f"does not correspond to the cached one ({self.prepared_data['protocol']})"
            )

        self.specifications = Specifications(
            problem=Problem.REPRESENTATION,
            resolution=Resolution.CHUNK,
            duration=self.duration,
            min_duration=self.min_duration,
            classes=sorted(self.prepared_data["train"]),
        )

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return [
            EqualErrorRate(compute_on_cpu=True, distances=False),
            BinaryAUROC(compute_on_cpu=True),
        ]
    def _sample_class_weighted_file_duration(self, rng):
        classes = list(self.specifications.classes)
        # Sample a class uniformly
        chosen_class = rng.choice(classes)
        
        # Sample a file weighted by duration
        chosen_file = rng.choices(
            self.prepared_data["train"][chosen_class],
            weights=[f["duration"] for f in self.prepared_data["train"][chosen_class]],
            k=1,
        )[0]
        
        return chosen_class, chosen_file

    def _sample_class_weighted_file_uniform(self, rng):
        classes = list(self.specifications.classes)
        num_files_per_class = [len(self.prepared_data["train"][klass]) for klass in classes]
        class_weights = num_files_per_class
        # Normalize weights
        total = sum(class_weights)
        class_probs = [w / total for w in class_weights]
        
        # Sample a class based on class_probs
        chosen_class = rng.choices(classes, weights=class_probs, k=1)[0]
        
        # Uniformly sample a file from the chosen class
        chosen_file = rng.choice(self.prepared_data["train"][chosen_class])
        
        return chosen_class, chosen_file

    def _initialize_file_pool(self):
        """Initialize a pool of files for sampling without replacement."""
        self.file_pool = {}
        for klass, files in self.prepared_data["train"].items():
            self.file_pool[klass] = files.copy()
        
    def _sample_file_weighted_no_replacement(self, rng, klass):
        if klass not in self.file_pool or not self.file_pool[klass]:
            # No files left to sample from this class
            return None, None

        # Extract files and their durations for the given class
        available_files = self.file_pool[klass]

        if not available_files:
            # All files from this class have been sampled
            return None, None

        # Sample a file based on duration weights without replacement
        chosen_file = rng.choices(available_files, k=1)[0]

        # Remove the chosen file from the pool to prevent reselection
        self.file_pool[klass].remove(chosen_file)

        return klass, chosen_file

    def train__iter__(self):
        """Iterate over training samples for one epoch

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: int
            Speaker index.
        """

        # Create worker-specific random number generator
        rng = create_rng_for_worker(self.model)

        # Select batch-wise duration at random
        batch_duration = rng.uniform(self.min_duration, self.duration)

        # Determine number of batches per epoch
        num_batches = self.train__len__()

        # Initialize file pool if using FILE_WEIGHTED_NO_REPLACEMENT
        if self.sampling_mode == SamplingMode.FILE_WEIGHTED_NO_REPLACEMENT and not hasattr(self, "file_pool"):
            print("Initializing file pool")
            self._initialize_file_pool()

        while True:
            # Sample a set of unique classes for the batch
            if self.sampling_mode == SamplingMode.CLASS_WEIGHTED_FILE_UNIFORM:
                # Sample classes weighted by the number of files they have
                classes = list(self.specifications.classes)
                num_files_per_class = [len(self.prepared_data["train"][klass]) for klass in classes]
                class_weights = num_files_per_class
                total = sum(class_weights)
                class_probs = [w / total for w in class_weights]
                # Use NumPy to sample without replacement with probabilities
                sampled_classes = list(np.random.choice(
                    classes,
                    size=self.num_classes_per_batch,
                    replace=False,
                    p=class_probs
                ))
            elif self.sampling_mode == SamplingMode.FILE_WEIGHTED_NO_REPLACEMENT:
                # Sample classes weighted by the total duration of their available files
                available_classes = [klass for klass, files in self.file_pool.items() if len(files) > 0]
                if len(available_classes) < self.num_classes_per_batch:
                    self._initialize_file_pool()
                    available_classes = [klass for klass, files in self.file_pool.items() if len(files) > 0]
                num_files_per_class = [len(self.file_pool[klass]) for klass in available_classes]
                class_weights = num_files_per_class
                total = sum(class_weights)
                class_probs = [w / total for w in class_weights]
                sampled_classes = list(np.random.choice(
                    available_classes,
                    size=self.num_classes_per_batch,
                    replace=False,
                    p=class_probs
                ))
            else:
                # SamplingMode.RANDOM_CLASS_WEIGHTED_FILE_DURATION: sample classes uniformly
                sampled_classes = rng.sample(self.specifications.classes, self.num_classes_per_batch)

            # Initialize batch_samples list
            batch_samples = []

            for klass in sampled_classes:
                if self.sampling_mode == SamplingMode.CLASS_WEIGHTED_FILE_UNIFORM:
                    # Sample a file uniformly from the class
                    chosen_file = rng.choice(self.prepared_data["train"][klass])
                elif self.sampling_mode == SamplingMode.FILE_WEIGHTED_NO_REPLACEMENT:
                    # Sample a file weighted by duration without replacement
                    chosen_class, chosen_file = self._sample_file_weighted_no_replacement(rng, klass)
                    if chosen_file is None:
                        raise ValueError(f"No files left to sample from class {klass}")
                else:
                    # SamplingMode.RANDOM_CLASS_WEIGHTED_FILE_DURATION: sample file weighted by duration
                    chosen_file = rng.choices(
                        self.prepared_data["train"][klass],
                        weights=[f["duration"] for f in self.prepared_data["train"][klass]],
                        k=1
                    )[0]

                # Sample a speech turn
                speech_turn, *_ = rng.choices(
                    chosen_file["speech_turns"],
                    weights=[s.duration for s in chosen_file["speech_turns"]],
                    k=1,
                )

                # Handle padding or cropping
                if speech_turn.duration < batch_duration:
                    X, _ = self.model.audio.crop(chosen_file, speech_turn)
                    num_missing_frames = (
                        math.floor(batch_duration * self.model.audio.sample_rate)
                        - X.shape[1]
                    )
                    left_pad = rng.randint(0, num_missing_frames)
                    X = F.pad(X, (left_pad, num_missing_frames - left_pad))
                else:
                    start_time = rng.uniform(
                        speech_turn.start, speech_turn.end - batch_duration
                    )
                    chunk = Segment(start_time, start_time + batch_duration)

                    X, _ = self.model.audio.crop(
                        chosen_file,
                        chunk,
                        duration=batch_duration,
                    )

                y = self.specifications.classes.index(klass)

                batch_samples.append({"X": X, "y": y})

            # Yield all samples in the batch
            for sample in batch_samples:
                yield sample

    def train__len__(self):
        if self.sampling_mode == SamplingMode.FILE_WEIGHTED_NO_REPLACEMENT:
            # Total number of files across all classes
            total_files = sum(len(files) for files in self.prepared_data["train"].values())
            # Calculate number of batches per epoch
            return math.ceil(total_files /1.2)
        else:
            # Original computation for other sampling modes
            total_duration = sum( datum["duration"] for data in self.prepared_data["train"].values() for datum in data )
            avg_chunk_duration = 0.5 * (self.min_duration + self.duration)
            return max(self.batch_size, math.ceil(total_duration / avg_chunk_duration))


    def collate_fn(self, batch, stage="train"):
        collated = default_collate(batch)

        if stage == "train":
            self.augmentation.train(mode=True)
            augmented = self.augmentation(
                samples=collated["X"],
                sample_rate=self.model.hparams.sample_rate,
            )
            collated["X"] = augmented.samples

        return collated

    def training_step(self, batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        loss = self.model.loss_func(self.model(X), y)

        if not self.model.automatic_optimization:

            wavlm_opt, other_opt = self.model.optimizers()

            wavlm_opt.zero_grad()
            other_opt.zero_grad()

            self.model.manual_backward(loss)

            wavlm_opt.step()
            other_opt.step()

        # skip batch if something went wrong for some reason
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

    def prepare_validation(self, prepared_dict: Dict):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            prepared_dict["validation"] = []
            for trial in self.protocol.development_trial():
                prepared_dict["validation"].append(
                    {
                        "reference": trial["reference"],
                        "file1": trial["file1"]["audio"],
                        "file2": trial["file2"]["audio"],
                    }
                )

    def val__getitem__(self, idx):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            trial = self.prepared_data["validation"][idx]

            data = dict()
            for idx in [1, 2]:
                file = trial[f"file{idx:d}"]
                duration = self.model.audio.get_duration(file)
                if duration > self.duration:
                    middle = Segment(
                        0.5 * duration - 0.5 * self.duration,
                        0.5 * duration + 0.5 * self.duration,
                    )
                    X, _ = self.model.audio.crop(file, middle, duration=self.duration)
                else:
                    X, _ = self.model.audio(file)
                    num_missing_frames = (
                        math.floor(self.duration * self.model.audio.sample_rate)
                        - X.shape[1]
                    )
                    X = F.pad(X, (0, num_missing_frames))
                data[f"X{idx:d}"] = X
            data["y"] = trial["reference"]

            return data

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            pass

    def val__len__(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            # breakpoint()
            return len(self.prepared_data["validation"])

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            return 0

    def validation_step(self, batch, batch_idx: int):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            with torch.no_grad():
                emb1 = self.model(batch["X1"]).detach()
                emb2 = self.model(batch["X2"]).detach()
                y_pred = F.cosine_similarity(emb1, emb2)

            y_true = batch["y"]
            self.model.validation_metric(y_pred, y_true)

            self.model.log_dict(
                self.model.validation_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
