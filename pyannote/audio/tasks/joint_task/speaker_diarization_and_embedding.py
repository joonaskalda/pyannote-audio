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

from collections import defaultdict
import itertools
from pathlib import Path
import random
import warnings
from tempfile import mkstemp
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
)
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_metric_learning.losses import ArcFaceLoss
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
import torch.nn as nn

from scipy.spatial.distance import cdist

from pyannote.audio import Inference
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Resolution, Specifications, get_dtype
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.audio.pipelines.clustering import KMeansClustering, OracleClustering
from pyannote.audio.pipelines.utils import SpeakerDiarizationMixin

from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)

from torchmetrics import MetricCollection

Subtask = Literal["diarization", "embedding"]

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)
Subtasks = list(Subtask.__args__)


class JointSpeakerDiarizationAndEmbedding(SpeakerDiarization):
    """Joint speaker diarization and embedding task

    Usage
    -----
    load a meta protocol containing both diarization (e.g. X.SpeakerDiarization.Pretraining)
    and verification (e.g. VoxCeleb.SpeakerVerification.VoxCeleb) datasets
    >>> from pyannote.database import registry
    >>> protocol = registry.get_protocol(...)

    instantiate task
    >>> task = JointSpeakerDiarizationAndEmbedding(protocol)

    instantiate multi-task model
    >>> model = JointSpeakerDiarizationAndEmbeddingModel()
    >>> model.task = task

    train as usual...

    """

    def __init__(
        self,
        protocol,
        duration: float = 5.0,
        max_speakers_per_chunk: int = 3,
        max_speakers_per_frame: int = 2,
        weigh_by_cardinality: bool = False,
        batch_size: int = 32,
        dia_task_rate: float = 0.5,
        num_workers: int = None,
        pin_memory: bool = False,
        margin: float = 28.6,
        scale: float = 64.0,
        alpha: float = 0.5,
        augmentation: BaseWaveformTransform = None,
        cache: Optional[Union[str, None]] = None,
    ) -> None:
        """TODO Add docstring"""
        super().__init__(
            protocol,
            duration=duration,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            weigh_by_cardinality=weigh_by_cardinality,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            cache=cache,
        )

        self.num_dia_samples = int(batch_size * dia_task_rate)
        self.margin = margin
        self.scale = scale
        self.alpha = alpha
        # keep track of the use of database available in the meta protocol
        # * embedding databases are those with global speaker label scope
        # * diarization databases are those with file or database speaker label scope
        self.embedding_files_id = []

        self.validation_metrics = MetricCollection(
            {
                "DiarizationErrorRate": DiarizationErrorRate(0.5),
                "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
                "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
                "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
            }
        )

        self.oracle_validation_metrics = self.validation_metrics.clone(prefix="Oracle")

    def prepare_data(self):
        """Use this to prepare data from task protocol

        Notes
        -----
        Called only once on the main process (and only on it), for global_rank 0.

        After this method is called, the task should have a `prepared_data` attribute
        with the following dictionary structure:

        prepared_data = {
            'protocol': name of the protocol
            'audio-path': array of N paths to audio
            'audio-metadata': array of N audio infos such as audio subset, scope and database
            'audio-info': array of N audio torchaudio.info struct
            'audio-encoding': array of N audio encodings
            'audio-annotated': array of N annotated duration (usually equals file duration but might be shorter if file is not fully annotated)
            'annotations-regions': array of M annotated regions
            'annotations-segments': array of M' annotated segments
            'metadata-values': dict of lists of values for subset, scope and database
            'metadata-`database-name`-labels': array of `database-name` labels. Each database with "database" scope labels has it own array.
            'metadata-labels': array of global scope labels
        }

        """

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

        # list of possible values for each metadata key
        # (will become .prepared_data[""])
        metadata_unique_values = defaultdict(list)
        metadata_unique_values["subset"] = Subsets
        metadata_unique_values["scope"] = Scopes

        audios = list()  # list of path to audio files
        audio_infos = list()
        audio_encodings = list()
        metadata = list()  # list of metadata

        annotated_duration = list()  # total duration of annotated regions (per file)
        annotated_regions = list()  # annotated regions
        annotations = list()  # actual annotations
        unique_labels = list()
        database_unique_labels = {}

        if self.has_validation:
            files_iter = itertools.chain(
                zip(itertools.repeat("train"), self.protocol.train()),
                zip(itertools.repeat("development"), self.protocol.development()),
            )
        else:
            files_iter = zip(itertools.repeat("train"), self.protocol.train())

        for file_id, (subset, file) in enumerate(files_iter):
            # gather metadata and update metadata_unique_values so that each metadatum
            # (e.g. source database or label) is represented by an integer.
            metadatum = dict()

            # keep track of source database and subset (train, development, or test)
            if file["database"] not in metadata_unique_values["database"]:
                metadata_unique_values["database"].append(file["database"])
            metadatum["database"] = metadata_unique_values["database"].index(
                file["database"]
            )

            metadatum["subset"] = Subsets.index(subset)

            # keep track of label scope (file, database, or global)
            metadatum["scope"] = Scopes.index(file["scope"])

            remaining_metadata_keys = set(file) - set(
                [
                    "uri",
                    "database",
                    "subset",
                    "audio",
                    "torchaudio.info",
                    "scope",
                    "classes",
                    "annotation",
                    "annotated",
                ]
            )

            # keep track of any other (integer or string) metadata provided by the protocol
            # (e.g. a "domain" key for domain-adversarial training)
            for key in remaining_metadata_keys:
                value = file[key]

                if isinstance(value, str):
                    if value not in metadata_unique_values[key]:
                        metadata_unique_values[key].append(value)
                    metadatum[key] = metadata_unique_values[key].index(value)

                elif isinstance(value, int):
                    metadatum[key] = value

                else:
                    warnings.warn(
                        f"Ignoring '{key}' metadata because of its type ({type(value)}). Only str and int are supported for now.",
                        category=UserWarning,
                    )

            metadata.append(metadatum)

            # reset list of file-scoped labels
            file_unique_labels = list()

            # path to audio file
            audios.append(str(file["audio"]))

            # audio info
            audio_info = file["torchaudio.info"]
            audio_infos.append(
                (
                    audio_info.sample_rate,  # sample rate
                    audio_info.num_frames,  # number of frames
                    audio_info.num_channels,  # number of channels
                    audio_info.bits_per_sample,  # bits per sample
                )
            )
            audio_encodings.append(audio_info.encoding)  # encoding

            # annotated regions and duration
            _annotated_duration = 0.0
            for segment in file["annotated"]:
                # skip annotated regions that are shorter than training chunk duration
                # if segment.duration < self.duration:
                # continue

                # append annotated region
                annotated_region = (
                    file_id,
                    segment.duration,
                    segment.start,
                )
                annotated_regions.append(annotated_region)

                # increment annotated duration
                _annotated_duration += segment.duration

            # append annotated duration
            annotated_duration.append(_annotated_duration)

            # annotations
            for segment, _, label in file["annotation"].itertracks(yield_label=True):
                # "scope" is provided by speaker diarization protocols to indicate
                # whether speaker labels are local to the file ('file'), consistent across
                # all files in a database ('database'), or globally consistent ('global')

                # 0 = 'file' / 1 = 'database' / 2 = 'global'
                scope = Scopes.index(file["scope"])

                # update list of file-scope labels
                if label not in file_unique_labels:
                    file_unique_labels.append(label)
                # and convert label to its (file-scope) index
                file_label_idx = file_unique_labels.index(label)

                database_label_idx = global_label_idx = -1

                if scope > 0:  # 'database' or 'global'
                    # update list of database-scope labels
                    database = file["database"]
                    if database not in database_unique_labels:
                        database_unique_labels[database] = []
                    if label not in database_unique_labels[database]:
                        database_unique_labels[database].append(label)

                    # and convert label to its (database-scope) index
                    database_label_idx = database_unique_labels[database].index(label)

                if scope > 1:  # 'global'
                    # update list of global-scope labels
                    if label not in unique_labels:
                        unique_labels.append(label)
                    # and convert label to its (global-scope) index
                    global_label_idx = unique_labels.index(label)

                annotations.append(
                    (
                        file_id,  # index of file
                        segment.start,  # start time
                        segment.end,  # end time
                        file_label_idx,  # file-scope label index
                        database_label_idx,  # database-scope label index
                        global_label_idx,  # global-scope index
                    )
                )

        # since not all metadata keys are present in all files, fallback to -1 when a key is missing
        metadata = [
            tuple(metadatum.get(key, -1) for key in metadata_unique_values)
            for metadatum in metadata
        ]
        metadata_dtype = [
            (key, get_dtype(max(m[i] for m in metadata)))
            for i, key in enumerate(metadata_unique_values)
        ]

        # turn list of files metadata into a single numpy array
        # TODO: improve using https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
        info_dtype = [
            (
                "sample_rate",
                get_dtype(max(ai[0] for ai in audio_infos)),
            ),
            (
                "num_frames",
                get_dtype(max(ai[1] for ai in audio_infos)),
            ),
            ("num_channels", "B"),
            ("bits_per_sample", "B"),
        ]

        # turn list of annotated regions into a single numpy array
        region_dtype = [
            (
                "file_id",
                get_dtype(max(ar[0] for ar in annotated_regions)),
            ),
            ("duration", "f"),
            ("start", "f"),
        ]

        # turn list of annotations into a single numpy array
        segment_dtype = [
            (
                "file_id",
                get_dtype(max(a[0] for a in annotations)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("file_label_idx", get_dtype(max(a[3] for a in annotations))),
            ("database_label_idx", get_dtype(max(a[4] for a in annotations))),
            ("global_label_idx", get_dtype(max(a[5] for a in annotations))),
        ]

        # save all protocol data in a dict
        prepared_data = {}

        # keep track of protocol name
        prepared_data["protocol"] = self.protocol.name

        prepared_data["audio-path"] = np.array(audios, dtype=np.str_)
        audios.clear()

        prepared_data["audio-metadata"] = np.array(metadata, dtype=metadata_dtype)
        metadata.clear()

        prepared_data["audio-info"] = np.array(audio_infos, dtype=info_dtype)
        audio_infos.clear()

        prepared_data["audio-encoding"] = np.array(audio_encodings, dtype=np.str_)
        audio_encodings.clear()

        prepared_data["audio-annotated"] = np.array(annotated_duration)
        annotated_duration.clear()

        prepared_data["annotations-regions"] = np.array(
            annotated_regions, dtype=region_dtype
        )
        annotated_regions.clear()

        prepared_data["annotations-segments"] = np.array(
            annotations, dtype=segment_dtype
        )
        annotations.clear()

        prepared_data["metadata-values"] = metadata_unique_values

        for database, labels in database_unique_labels.items():
            prepared_data[f"metadata-{database}-labels"] = np.array(
                labels, dtype=np.str_
            )
        database_unique_labels.clear()

        prepared_data["metadata-labels"] = np.array(unique_labels, dtype=np.str_)
        unique_labels.clear()

        if self.has_validation:
            self.prepare_validation(prepared_data)

        self.post_prepare_data(prepared_data)

        # save prepared data on the disk
        with open(self.cache, "wb") as cache_file:
            np.savez_compressed(cache_file, **prepared_data)

    def prepare_validation(self, prepared_data: Dict) -> None:
        """Each validation batch correspond to a part of a validation file"""
        validation_mask = prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "development"
        )
        prepared_data["validation-files"] = np.argwhere(validation_mask).reshape((-1,))

    def setup(self, stage="fit"):
        """Setup method

        Parameters
        ----------
        stage : {'fit', 'validate', 'test'}, optional
            Setup stage. Defaults to 'fit'.
        """

        super().setup()

        global_scope_mask = (
            self.prepared_data["annotations-segments"]["global_label_idx"] > -1
        )
        train_file_mask = self.prepared_data["audio-metadata"][
            "subset"
        ] == Subsets.index("train")
        train_file_ids = np.where(train_file_mask)[0]
        train_segment_mask = np.isin(
            self.prepared_data["annotations-segments"]["file_id"], train_file_ids
        )
        mask = train_segment_mask & global_scope_mask
        self.embedding_files_id = np.unique(
            self.prepared_data["annotations-segments"]["file_id"][mask]
        )
        embedding_classes = np.unique(
            self.prepared_data["annotations-segments"]["global_label_idx"][mask]
        )

        # if there is no file dedicated to the embedding task
        if self.alpha != 1.0 and len(embedding_classes) == 0:
            self.num_dia_samples = self.batch_size
            self.alpha = 1.0
            warnings.warn(
                "No class found for the speaker embedding task. Model will be trained on the speaker diarization task only."
            )

        if self.alpha != 0.0 and np.sum(global_scope_mask) == len(
            self.prepared_data["annotations-segments"]
        ):
            self.num_dia_samples = 0
            self.alpha = 0.0
            warnings.warn(
                "No segment found for the speaker diarization task. Model will be trained on the speaker embedding task only."
            )

        speaker_diarization = Specifications(
            duration=self.duration,
            resolution=Resolution.FRAME,
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            permutation_invariant=True,
            classes=[f"speaker{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
        )
        speaker_embedding = Specifications(
            duration=self.duration,
            resolution=Resolution.CHUNK,
            problem=Problem.REPRESENTATION,
            classes=embedding_classes,
        )
        self.specifications = (speaker_diarization, speaker_embedding)

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target as a SlidingWindowFeature instance where y.labels is
                   in meta.scope space.
            - `meta`:
                - `scope`: target scope (0: file, 1: database, 2: global)
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        # get label scope
        label_scope = Scopes[self.prepared_data["audio-metadata"][file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(
            file, chunk, duration=duration, mode="pad"
        )

        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start - half
        end_idx = np.round(end / step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start : end + 1, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(y, self.model.receptive_field, labels=labels)

        metadata = self.prepared_data["audio-metadata"][file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def draw_diarization_chunk(
        self,
        file_ids: np.ndarray,
        cum_prob_annotated_duration: np.ndarray,
        rng: random.Random,
        duration: float,
    ) -> tuple:
        """Sample one chunk for the diarization task

        Parameters
        ----------
        file_ids: np.ndarray
            array containing files id
        cum_prob_annotated_duration: np.ndarray
            array of the same size than file_ids array, containing probability
            to corresponding file to be drawn
        rng : random.Random
            Random number generator
        duration: float
            duration of the chunk to draw
        """
        # select one file at random (wiht probability proportional to its annotated duration)
        file_id = file_ids[cum_prob_annotated_duration.searchsorted(rng.random())]
        # find indices of annotated regions in this file
        annotated_region_indices = np.where(
            self.prepared_data["annotations-regions"]["file_id"] == file_id
        )[0]

        # turn annotated regions duration into a probability distribution
        cum_prob_annotaded_regions_duration = np.cumsum(
            self.prepared_data["annotations-regions"]["duration"][
                annotated_region_indices
            ]
            / np.sum(
                self.prepared_data["annotations-regions"]["duration"][
                    annotated_region_indices
                ]
            )
        )

        # seletect one annotated region at random (with probability proportional to its duration)
        annotated_region_index = annotated_region_indices[
            cum_prob_annotaded_regions_duration.searchsorted(rng.random())
        ]

        # select one chunk at random in this annotated region
        _, region_duration, start = self.prepared_data["annotations-regions"][
            annotated_region_index
        ]
        start_time = rng.uniform(start, start + region_duration - duration)

        return (file_id, start_time)

    def draw_embedding_chunk(self, class_id: int, duration: float) -> tuple:
        """Sample one chunk for the embedding task

        Parameters
        ----------
        class_id : int
            class ID in the task speficiations
        duration: float
            duration of the chunk to draw

        Return
        ------
        tuple:
            file_id:
                the file id to which the sampled chunk belongs
            start_time:
                start time of the sampled chunk
        """
        # get index of the current class in the order of original class list
        # get segments for current class
        class_segments_idx = (
            self.prepared_data["annotations-segments"]["global_label_idx"] == class_id
        )
        class_segments = self.prepared_data["annotations-segments"][class_segments_idx]

        # sample one segment from all the class segments:
        segments_duration = class_segments["end"] - class_segments["start"]
        segments_total_duration = np.sum(segments_duration)
        prob_segments = segments_duration / segments_total_duration
        segment = np.random.choice(class_segments, p=prob_segments)

        # sample chunk start time in order to intersect it with the sampled segment
        start_time = np.random.uniform(
            max(segment["start"] - duration, 0), segment["end"]
        )

        return (segment["file_id"], start_time)

    def train__iter__helper(self, rng: random.Random, **filters):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key : value} dict), filter training files so that
            only file such as file [key] == value are used for generating chunks

        Yields
        ------
        chunk : dict
        Training chunks
        """

        # indices of training files that matches domain filters
        training = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "train"
        )
        for key, value in filters.items():
            training &= self.prepared_data["audio-metadata"][key] == self.prepared_data[
                "metadata"
            ][key].index(value)
        file_ids = np.where(training)[0]
        # get the subset of embedding database files from training files
        embedding_files_ids = file_ids[np.isin(file_ids, self.embedding_files_id)]

        if self.num_dia_samples > 0:
            annotated_duration = self.prepared_data["audio-annotated"][file_ids]
            # set duration of files for the embedding part to zero, in order to not
            # drawn them for diarization part
            annotated_duration[embedding_files_ids] = 0.0

            cum_prob_annotated_duration = np.cumsum(
                annotated_duration / np.sum(annotated_duration)
            )

        duration = self.duration
        batch_size = self.batch_size

        # use original order for the first run on the shuffled classes list:
        emb_task_classes = self.specifications[Subtasks.index("embedding")].classes[:]

        sample_idx = 0
        embedding_class_idx = 0
        while True:
            if sample_idx < self.num_dia_samples:
                file_id, start_time = self.draw_diarization_chunk(
                    file_ids, cum_prob_annotated_duration, rng, duration
                )
            else:
                # shuffle embedding classes list and go through this shuffled list
                # to make sure to see all the speakers during training
                if embedding_class_idx == len(emb_task_classes):
                    rng.shuffle(emb_task_classes)
                    embedding_class_idx = 0
                klass = emb_task_classes[embedding_class_idx]
                embedding_class_idx += 1
                file_id, start_time = self.draw_embedding_chunk(klass, duration)

            sample = self.prepare_chunk(file_id, start_time, duration)
            sample_idx = (sample_idx + 1) % batch_size

            yield sample

    def train__iter__(self):
        """Iterate over trainig samples

        Yields
        ------
        dict:
            x: (time, channel)
                Audio chunks.
            task: "diarization" or "embedding"
            y: target speaker label for speaker embedding task,
                (frame, ) frame-level targets for speaker diarization task.
                Note that frame < time.
                `frame is infered automagically from the exemple model output`
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.model)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)
        else:
            # create
            subchunks = dict()
            for product in itertools.product(
                [self.prepared_data["metadata-values"][key] for key in balance]
            ):
                filters = {key: value for key, value in zip(balance, product)}
                subchunks[product] = self.train__iter__helper(rng, **filters)

        while True:
            # select one subchunck generator at random (with uniform probability)
            # so thath it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(subchunks)]

            # generate random chunk
            yield next(chunks)

    def val__getitem__(self, idx) -> Dict:
        """Validation items are generated so that all the chunks in a batch come from the same
        validation file. These chunks are extracted regularly over all the file, so that the first
        chunk start at the very beginning of the file, and the last chunk ends at the end of the file.
        Step between chunks depends of the file duration and the total batch duration. This step can
        be negative. In that case, chunks are overlapped.

        Parameters
        ----------
        idx: int
            item index. Note: this method may be incompatible with the use of sampler,
            as this method requires incremental idx starting from 0.

        Returns
        -------
        chunk: dict
            extracted chunk from current validation file. The first chunk contains annotation
            for the whole file.
        """

        file_idx = idx // self.batch_size
        chunk_idx = idx % self.batch_size

        file_id = self.prepared_data["validation-files"][file_idx]
        file = next(
            itertools.islice(self.protocol.development(), file_idx, file_idx + 1)
        )

        file_duration = file.get(
            "duration", Audio("downmix").get_duration(file["audio"])
        )
        start_time = chunk_idx * (
            (file_duration - self.duration) / (self.batch_size - 1)
        )

        chunk = self.prepare_chunk(file_id, start_time, self.duration)

        if chunk_idx == 0:
            chunk["annotation"] = file["annotation"]

        chunk["start_time"] = start_time

        return chunk

    def val__len__(self):
        """Return total length of validation, which is num_validation_files * batch_size"""
        return len(self.prepared_data["validation-files"]) * self.batch_size

    def collate_y(self, batch) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : list
            List of samples to collate.
            "y" field is expected to be a SlidingWindowFeature.

        Returns
        -------
        y : torch.Tensor
            Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
            If one chunk has more than `self.max_speakers_per_chunk` speakers, we keep
            the max_speakers_per_chunk most talkative ones. If it has less, we pad with
            zeros (artificial inactive speakers).
        """

        collated_y_dia = []
        collate_y_emb = []

        for b in batch:
            # diarization reference
            y_dia = b["y"].data
            labels = b["y"].labels
            num_speakers = len(labels)
            # embedding reference
            y_emb = np.full((self.max_speakers_per_chunk,), -1, dtype=int)

            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y_dia, axis=0), axis=0)
                # keep only the most talkative speakers
                y_dia = y_dia[:, indices[: self.max_speakers_per_chunk]]
                # TODO: we should also sort the speaker labels in the same way

                # if current chunck is for the embedding subtask
                if b["meta"]["scope"] > 1:
                    labels = np.array(labels)
                    y_emb = labels[indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                # create inactive speakers by zero padding
                y_dia = np.pad(
                    y_dia,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )
                if b["meta"]["scope"] > 1:
                    y_emb[:num_speakers] = labels[:]

            else:
                if b["meta"]["scope"] > 1:
                    y_emb[:num_speakers] = labels[:]

            collated_y_dia.append(y_dia)
            collate_y_emb.append(y_emb)

        return (
            torch.from_numpy(np.stack(collated_y_dia)),
            torch.from_numpy(np.stack(collate_y_emb)).squeeze(1),
        )

    def collate_fn(self, batch, stage="train"):
        """Collate function used for most segmentation tasks

        This function does the following:
        * stack waveforms into a (batch_size, num_channels, num_samples) tensor batch["X"])
        * apply augmentation when in "train" stage
        * convert targets into a (batch_size, num_frames, num_classes) tensor batch["y"]
        * collate any other keys that might be present in the batch using pytorch default_collate function

        Parameters
        ----------
        batch : list of dict
            List of training samples.

        Returns
        -------
        batch : dict
            Collated batch as {"X": torch.Tensor, "y": torch.Tensor} dict (train).
            Collated batch as {"X": torch.Tensor, "annotation": Annotation, "start_times": list} dict (validation)
        """

        # collate X
        collated_X = self.collate_X(batch)
        # collate y
        collated_y_dia, collate_y_emb = self.collate_y(batch)

        # collate metadata
        collated_meta = self.collate_meta(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y_dia.unsqueeze(1),
        )
        collated_batch = {
            "X": augmented.samples,
            "y_dia": augmented.targets.squeeze(1),
            "y_emb": collate_y_emb,
            "meta": collated_meta,
        }

        if stage == "val":
            collated_batch["annotation"] = batch[0]["annotation"]
            collated_batch["start_times"] = [b["start_time"] for b in batch]

        return collated_batch

    def setup_loss_func(self):
        self.model.arc_face_loss = ArcFaceLoss(
            len(self.specifications[Subtasks.index("embedding")].classes),
            self.model.hparams["embedding_dim"],
            margin=self.margin,
            scale=self.scale,
            weight_init_func=nn.init.xavier_normal_
        )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        # `clamp_min` is needed to set non-speech weight to 1.
        class_weight = (
            torch.clamp_min(self.model.powerset.cardinality, 1.0)
            if self.weigh_by_cardinality
            else None
        )
        seg_loss = nll_loss(
            permutated_prediction,
            torch.argmax(target, dim=-1),
            class_weight=class_weight,
            weight=weight,
        )

        return seg_loss

    def compute_diarization_loss(self, prediction, permutated_target):
        """Compute loss for the speaker diarization subtask

        Parameters
        ----------
        prediction : torch.Tensor
            speaker diarization output predicted by the model for the current batch.
            Shape of (batch_size, num_spk, num_frames)
        permutated_target: torch.Tensor
            permutated target for the current batch. Shape of (batch_size, num_spk, num_frames)

        Returns
        -------
        dia_loss : torch.Tensor
            Permutation-invariant diarization loss
        """

        # Compute segmentation loss
        dia_loss = self.segmentation_loss(prediction, permutated_target)
        self.model.log(
            "loss/train/dia",
            dia_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return dia_loss

    def compute_embedding_loss(self, emb_prediction, target_emb, valid_embs):
        """Compute loss for the speaker embeddings extraction subtask

        Parameters
        ----------
        emb_prediction : torch.Tensor
            speaker embeddings predicted by the model for the current batch.
            Shape of (batch_size * num_spk, embedding_dim)
        target_emb : torch.Tensor
            target embeddings for the current batch
            Shape of (batch_size * num_spk,)
        Returns
        -------
        emb_loss : torch.Tensor
            arcface loss for the current batch
        """

        # Get speaker representations from the embedding subtask
        embeddings = rearrange(emb_prediction, "b s e -> (b s) e")
        # Get corresponding target label
        targets = rearrange(target_emb, "b s -> (b s)")
        # compute loss only on global scope speaker embedding
        valid_embs = rearrange(valid_embs, "b s -> (b s)")
        # compute the loss
        emb_loss = self.model.arc_face_loss(
            embeddings[valid_embs, :], targets[valid_embs]
        )

        # skip batch if something went wrong for some reason
        if torch.isnan(emb_loss):
            return None

        self.model.log(
            "loss/train/arcface",
            emb_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return emb_loss

    def training_step(self, batch, batch_idx: int):
        """Compute loss for the joint task

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        # batch waveforms (batch_size, num_channels, num_samples)
        waveform = batch["X"]
        # batch diarization references (batch_size, num_channels, num_speakers)
        target_dia = batch["y_dia"]
        # batch embedding references (batch, num_speakers)
        target_emb = batch["y_emb"]

        # drop samples that contain too many speakers
        num_speakers = torch.sum(torch.any(target_dia, dim=1), dim=1)
        keep = num_speakers <= self.max_speakers_per_chunk

        target_dia = target_dia[keep]
        target_emb = target_emb[keep]
        waveform = waveform[keep]

        num_remaining_dia_samples = torch.sum(keep[: self.num_dia_samples])

        # corner case
        if not keep.any():
            return None

        # forward pass
        dia_prediction, emb_prediction = self.model(waveform)
        # (batch_size, num_frames, num_cls), (batch_size, num_spk, emb_size)

        # get the best permutation
        dia_multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target_dia, permut_map = permutate(dia_multilabel, target_dia)
        permutated_target_emb = target_emb[
            torch.arange(target_emb.shape[0]).unsqueeze(1), permut_map
        ]

        # an embedding is valid only if corresponding speaker is active in the diarization prediction and reference
        active_speaker_pred = torch.any(dia_multilabel > 0, dim=1)
        active_speaker_ref = torch.any(permutated_target_dia == 1, dim=1)
        valid_embs = torch.logical_and(active_speaker_pred, active_speaker_ref)[
            num_remaining_dia_samples:
        ]

        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target_dia.float()
        )

        dia_prediction = dia_prediction[:num_remaining_dia_samples]
        permutated_target_powerset = permutated_target_powerset[
            :num_remaining_dia_samples
        ]

        dia_loss = torch.tensor(0)
        # if batch contains diarization subtask chunks, then compute diarization loss on these chunks:
        if self.alpha != 0.0 and torch.any(keep[: self.num_dia_samples]):
            dia_loss = self.compute_diarization_loss(
                dia_prediction, permutated_target_powerset
            )

        emb_loss = torch.tensor(0)
        # if batch contains embedding subtask chunks, then compute embedding loss on these chunks:
        if self.alpha != 1.0 and torch.any(valid_embs):
            emb_prediction = emb_prediction[num_remaining_dia_samples:]
            permutated_target_emb = permutated_target_emb[num_remaining_dia_samples:]
            emb_loss = self.compute_embedding_loss(
                emb_prediction, permutated_target_emb, valid_embs
            )
            loss = self.alpha * dia_loss + (1 - self.alpha) * emb_loss
        else:
            loss = self.alpha * dia_loss

        return {"loss": loss}

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        clusters: np.ndarray,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        clusters : (num_chunks, num_speakers) array
            Output of clustering step.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, _ = segmentations.data.shape
        num_clusters = np.max(clusters) + 1
        clustered_segmentations = np.zeros((num_chunks, num_frames, num_clusters))

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):
            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return clustered_segmentations

    # def aggregate(
    #     self, segmentations: SlidingWindowFeature, pad_duration: float
    # ) -> SlidingWindowFeature:
    #     """ "Aggregate segmentation chunks over time with padding between
    #     each chunk.

    #     Parameters
    #     ----------
    #     segmentations: SlidingWindowFeature
    #         unaggragated segmentation chunks. Shape is (num_chunks, num_frames, num_speakers).
    #     pad_duration: float
    #         padding duration between two consecutive segmentation chunks.
    #         In case pad_duration is < 0. (overlapped segmentation chunks),
    #         `Inference.aggregate()` is used.

    #     Returns
    #     -------
    #     segmentation: SlidingWindowFeature
    #         aggregated segmentation. Shape is (num_frames', num_speakers).
    #     """

    #     num_chunks, num_frames, num_speakers = segmentations.data.shape
    #     frame_duration = segmentations.sliding_window.duration / num_frames

    #     window = SlidingWindow(step=frame_duration, duration=frame_duration)

    #     if num_chunks == 1:
    #         return SlidingWindowFeature(segmentations[0], window)

    #     # if segmentation chunks are overlaped
    #     if pad_duration < 0.0:
    #         return Inference.aggregate(segmentations, window)

    #     num_padding_frames = np.round(pad_duration / frame_duration).astype(np.uint32)
    #     aggregated_segmentation = segmentations[0]

    #     for chunk_segmentation in segmentations[1:]:
    #         padding = np.zeros((num_padding_frames, num_speakers))
    #         aggregated_segmentation = np.concatenate(
    #             (aggregated_segmentation, padding, chunk_segmentation)
    #         )

    #     return SlidingWindowFeature(aggregated_segmentation.astype(np.int8), window)

    # def to_diarization(
    #     self,
    #     segmentations: SlidingWindowFeature,
    #     pad_duration: float = 0.0,
    # ) -> SlidingWindowFeature:
    #     """Build diarization out of preprocessed segmentation

    #     Parameters
    #     ----------
    #     segmentations : SlidingWindowFeature
    #         (num_chunks, num_frames, num_speakers)-shaped segmentations
    #     pad_duration: float,
    #         padding duration between two consecutive segmentation chunks.
    #         Can be negative (overlapped chunks).

    #     Returns
    #     -------
    #     discrete_diarization : SlidingWindowFeature
    #         Discrete (0s and 1s) diarization. Shape is (num_frames', num_speakers')
    #     """

    #     activations = self.aggregate(segmentations, pad_duration=pad_duration)
    #     # shape: (num_frames, num_speakers)
    #     _, num_speakers = activations.data.shape

    #     count = np.sum(activations, axis=1, keepdims=True)
    #     # shape: (num_frames, 1)

    #     max_speakers_per_frame = np.max(count.data)
    #     if num_speakers < max_speakers_per_frame:
    #         activations.data = np.pad(
    #             activations.data, ((0, 0), (0, max_speakers_per_frame - num_speakers))
    #         )

    #     extent = activations.extent & count.extent
    #     activations = activations.crop(extent, return_data=False)
    #     count = count.crop(extent, return_data=False)

    #     sorted_speakers = np.argsort(-activations, axis=-1)
    #     binary = np.zeros_like(activations.data)

    #     for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
    #         for i in range(c.item()):
    #             binary[t, speakers[i]] = 1.0

    #     return SlidingWindowFeature(binary, activations.sliding_window)

    def compute_metrics(
        self,
        discretized_reference,
        prediction: Tuple[SlidingWindowFeature, np.ndarray],
        oracle_mode: bool,
    ) -> None:
        """Compute (oracle) Diarization Error Rate at file level 
        given the reference and hypothesis.
        DER is only computed on parts of the validation file 
        that were used to build validation batch.

        Parameters
        ----------
        discretized_reference: np.ndarray
            cropped file's discretized reference matching parts 
            of the validation file used to build current batch
        prediction: (pyannote.core.SlidingWindowFeature, np.ndarray)
            tuple containing unclustered segmentation chunks 
            and clusters from the clustering step
        oracle_mode: boolean
            Whether to compute DER or oracle DER
        """
        binarized_segmentations, clusters = prediction

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        # shape: (num_chunks, num_speakers)
        clusters[inactive_speakers] = -2

        clustered_segmentations = self.reconstruct(binarized_segmentations, clusters)
        # shape: (num_chunks, num_frames, num_speakers)

        clustered_segmentations = torch.from_numpy(clustered_segmentations.data)
        hypothesis = rearrange(clustered_segmentations, "c f s -> s (c f)")
        # shape: (num_speakers, num_chunks * num_frames)

        reference = torch.from_numpy(discretized_reference.T)
        # shape: (num_speakers, num_chunks * num_frames)

        # calculate and log metrics
        name = "oracle_validation_metrics" if oracle_mode else "validation_metrics"
        metrics = getattr(self, name)
        outputs = metrics(hypothesis.unsqueeze(0), reference.unsqueeze(0))
        self.model.log_dict(
            outputs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx: int):
        """Validation step consists of applying a diarization pipeline
        on a validation file to compute file-level Diarization Error Rate (DER)
        and Oracle Diarization Error Rate (ODER).

        Parameters
        ----------
        batch : dict of torch.Tensor
            current batch. All chunks come from the same
            file and are in chronological order
        batch_idx: int
            Batch index.
        """

        # get reference
        reference = batch["annotation"]
        num_speakers = len(reference.labels())

        start_times = batch["start_times"]

        file_id = batch["meta"]["file"][0]
        file = self.get_file(file_id)
        # needed by oracle clustering
        file["annotation"] = reference

        assert reference.uri in file["audio"]

        resolution = self.model.receptive_field

        # get discretized reference for current file
        discretized_segments = []
        num_frames = int(
            self.model.num_frames(self.model.hparams["sample_rate"] * self.duration)
        )
        for start_time in start_times:
            discretized_segment = reference.discretize(
                support=Segment(start_time, start_time + self.duration),
                resolution=resolution,
                labels=reference.labels(),
            )
            discretized_segments.append(discretized_segment.data[:num_frames])
        discretized_reference = np.concatenate(discretized_segments)
        # shape: (num_chunks * num_frames, num_speakers)

        waveform = batch["X"]
        # shape: (num_chunks, num_channels, local_num_samples)

        # segmentation + embeddings extraction step
        segmentations, embeddings = self.model(waveform)
        # shapes: (num_chunks, num_frames, powerset_classes), (num_chunks, local_num_speakers, embed_dim)

        if self.batch_size > 1:
            step = batch["start_times"][1] - batch["start_times"][0]
        else:
            step = self.duration

        # convert from powerset segmentations to multilabel segmentations
        binarized_segmentations = self.model.powerset.to_multilabel(segmentations)

        # gradient is uneeded here, so we can detach tensors from the gradient graph
        binarized_segmentations = binarized_segmentations.cpu().detach().numpy()
        embeddings = embeddings.cpu().detach().numpy()

        binarized_segmentations = SlidingWindowFeature(
            binarized_segmentations,
            SlidingWindow(
                start=batch["start_times"][0], duration=self.duration, step=step
            ),
        )

        # compute file-wise diarization error rate
        clustering = KMeansClustering()
        clusters, _, _ = clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
        )
        der = self.compute_metrics(
            discretized_reference=discretized_reference,
            prediction=(binarized_segmentations, clusters),
            oracle_mode=False,
        )

        # compute file-wise oracle diarization error rate
        oracle_clustering = OracleClustering()
        oracle_clusters, _, _ = oracle_clustering(
            segmentations=binarized_segmentations,
            file=file,
            frames=resolution.step,
        )
        oder = self.compute_metrics(
            discretized_reference=discretized_reference,
            prediction=(binarized_segmentations, oracle_clusters),
            oracle_mode=True,
        )

        return None

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components for diarization subtask,
        and equal error rate for the embedding part
        """
        return {}
