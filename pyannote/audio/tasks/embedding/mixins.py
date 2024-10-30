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
        self.drop_last = False
        self.seed = 42
        
        self.sampling_mode = "classes_weighted_file_uniform"
        if self.sampling_mode == "espnet":
            self.precompute_batches()

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return [
            EqualErrorRate(compute_on_cpu=True, distances=False),
            BinaryAUROC(compute_on_cpu=True),
        ]
        
    def precompute_batches(self):
        """Precompute batches to mimic the first DataLoader's behavior."""
        rng = np.random.RandomState(self.seed)

        # Prepare list of all utterances per speaker
        self.speaker_to_utterances = {}
        for speaker in self.specifications.classes:
            utterances = []
            for file_info in self.prepared_data["train"][speaker]:
                for speech_turn in file_info["speech_turns"]:
                    utterances.append({
                        "file": file_info,
                        "segment": speech_turn
                    })
            rng.shuffle(utterances)
            self.speaker_to_utterances[speaker] = utterances
        # breakpoint()
        # Compute total number of batches
        total_utterances = sum(len(utt_list) for utt_list in self.speaker_to_utterances.values())
        self.total_batches = total_utterances // self.batch_size

        # Precompute batches
        self.precomputed_batches = []
        speakers = list(self.speaker_to_utterances.keys())
        rng.shuffle(speakers)

        # Flatten utterances with speaker labels
        utterance_queue = []
        for speaker in speakers:
            utterances = self.speaker_to_utterances[speaker]
            for utt in utterances:
                utterance_queue.append((speaker, utt))
        rng.shuffle(utterance_queue)

        current_batch = []
        current_speakers_in_batch = set()
        for speaker, utt in utterance_queue:
            if speaker in current_speakers_in_batch:
                continue  # Skip if speaker already in batch
            current_batch.append((speaker, utt))
            current_speakers_in_batch.add(speaker)

            if len(current_batch) == self.batch_size:
                self.precomputed_batches.append(current_batch)
                current_batch = []
                current_speakers_in_batch = set()

        # Handle the last batch
        if current_batch and not self.drop_last:
            self.precomputed_batches.append(current_batch)

        self.seed+=1
        # breakpoint()

    def _sample_original(self, rng):
        """Original sampling method from Version 1."""
        classes = list(self.specifications.classes)
        rng.shuffle(classes)
        for klass in classes:
            y = self.specifications.classes.index(klass)
            for _ in range(self.num_chunks_per_class):
                file = rng.choices(
                    self.prepared_data["train"][klass],
                    weights=[f["duration"] for f in self.prepared_data["train"][klass]],
                    k=1,
                )[0]
                speech_turn = rng.choices(
                    file["speech_turns"],
                    weights=[s.duration for s in file["speech_turns"]],
                    k=1,
                )[0]

                if speech_turn.duration < self.min_duration:
                    X, _ = self.model.audio.crop(file, speech_turn)
                    num_missing_frames = (
                        math.floor(self.min_duration * self.model.audio.sample_rate)
                        - X.shape[1]
                    )
                    left_pad = rng.randint(0, num_missing_frames)
                    X = F.pad(X, (left_pad, num_missing_frames - left_pad))
                else:
                    start_time = rng.uniform(
                        speech_turn.start, speech_turn.end - self.min_duration
                    )
                    chunk = Segment(start_time, start_time + self.min_duration)
                    X, _ = self.model.audio.crop(file, chunk, duration=self.min_duration)
                yield {"X": X, "y": y}

    def train__iter__(self):
        """Iterate over precomputed training batches."""
        if self.sampling_mode == "original":
            rng = create_rng_for_worker(self.model)

            classes = list(self.specifications.classes)

            # select batch-wise duration at random
            batch_duration = rng.uniform(self.min_duration, self.duration)
            num_samples = 0

            while True:
                # shuffle classes so that we don't always have the same
                # groups of classes in a batch (which might be especially
                # problematic for contrast-based losses like contrastive
                # or triplet loss.
                rng.shuffle(classes)

                for klass in classes:
                    # class index in original sorted order
                    y = self.specifications.classes.index(klass)

                    # multiple chunks per class
                    for _ in range(self.num_chunks_per_class):
                        # select one file at random (with probability proportional to its class duration)
                        file, *_ = rng.choices(
                            self.prepared_data["train"][klass],
                            weights=[
                                f["duration"] for f in self.prepared_data["train"][klass]
                            ],
                            k=1,
                        )

                        # select one speech turn at random (with probability proportional to its duration)
                        speech_turn, *_ = rng.choices(
                            file["speech_turns"],
                            weights=[s.duration for s in file["speech_turns"]],
                            k=1,
                        )

                        # if speech turn is too short, pad with zeros
                        # TODO: handle this corner case with recently added mode="pad" option to audio.crop
                        if speech_turn.duration < batch_duration:
                            X, _ = self.model.audio.crop(file, speech_turn)
                            num_missing_frames = (
                                math.floor(batch_duration * self.model.audio.sample_rate)
                                - X.shape[1]
                            )
                            left_pad = rng.randint(0, num_missing_frames)
                            X = F.pad(X, (left_pad, num_missing_frames - left_pad))

                        # if it is long enough, select chunk at random
                        else:
                            start_time = rng.uniform(
                                speech_turn.start, speech_turn.end - batch_duration
                            )
                            chunk = Segment(start_time, start_time + batch_duration)

                            X, _ = self.model.audio.crop(
                                file,
                                chunk,
                                duration=batch_duration,
                            )

                        yield {"X": X, "y": y}

                        num_samples += 1
                        if num_samples == self.batch_size:
                            batch_duration = rng.uniform(self.min_duration, self.duration)
                            num_samples = 0  # Use 'yield from' instead of 'return''

        elif self.sampling_mode == "classes_weighted_file_uniform":
            rng = create_rng_for_worker(self.model)

            classes = list(self.specifications.classes)

            # select batch-wise duration at random
            batch_duration = rng.uniform(self.min_duration, self.duration)
            num_samples = 0

            while True:
            # Sample a set of unique classes for the batch
                
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
                for klass in sampled_classes:
                    y = self.specifications.classes.index(klass)
                    file = rng.choice(self.prepared_data["train"][klass])
                # select one speech turn at random (with probability proportional to its duration)
                    speech_turn, *_ = rng.choices(
                        file["speech_turns"],
                        weights=[s.duration for s in file["speech_turns"]],
                        k=1,
                    )

                    # if speech turn is too short, pad with zeros
                    # TODO: handle this corner case with recently added mode="pad" option to audio.crop
                    if speech_turn.duration < batch_duration:
                        X, _ = self.model.audio.crop(file, speech_turn)
                        num_missing_frames = (
                            math.floor(batch_duration * self.model.audio.sample_rate)
                            - X.shape[1]
                        )
                        left_pad = rng.randint(0, num_missing_frames)
                        X = F.pad(X, (left_pad, num_missing_frames - left_pad))

                    # if it is long enough, select chunk at random
                    else:
                        start_time = rng.uniform(
                            speech_turn.start, speech_turn.end - batch_duration
                        )
                        chunk = Segment(start_time, start_time + batch_duration)

                        X, _ = self.model.audio.crop(
                            file,
                            chunk,
                            duration=batch_duration,
                        )

                    yield {"X": X, "y": y}

                    num_samples += 1
                    if num_samples == self.batch_size:
                        batch_duration = rng.uniform(self.min_duration, self.duration)
                        num_samples = 0  # Use 'yield from' instead of 'return''

        elif self.sampling_mode == "espnet":
            if self.seed > 42:
                print("Seed is greater than 42")
                self.precompute_batches()
            
            # Existing code for other sampling modes...
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                iter_batches = self.precomputed_batches
            else:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                iter_batches = self.precomputed_batches[worker_id::num_workers]
            
            for batch in iter_batches:
                batch_samples = []
                for speaker, utt_info in batch:
                    file = utt_info["file"]
                    segment = utt_info["segment"]
                    # Load and process the audio segment
                    X, _ = self.model.audio.crop(file, segment, duration=self.duration)
                    y = self.specifications.classes.index(speaker)
                    batch_samples.append({"X": X, "y": y})
            
                for sample in batch_samples:
                    yield sample

    def train__len__(self):
        if self.sampling_mode == "espnet":
             return len(self.precomputed_batches) * self.batch_size
        # if self.sampling_mode == "original":
        duration = sum(
            datum["duration"]
            for data in self.prepared_data["train"].values()
            for datum in data
        )
        avg_chunk_duration = 0.5 * (self.min_duration + self.duration)
        return max(self.batch_size, math.ceil(duration / avg_chunk_duration))
    

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
