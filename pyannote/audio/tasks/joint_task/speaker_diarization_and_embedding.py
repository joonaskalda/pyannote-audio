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
from pyannote.audio.torchmetrics.classification import EqualErrorRate
from torchmetrics.classification import BinaryAUROC
from functools import cached_property, partial
from torch.utils.data import DataLoader, Dataset, IterableDataset


from copy import deepcopy
from collections import defaultdict
import itertools
from pathlib import Path
import random
import warnings
from tempfile import mkstemp
from typing import Dict, Literal, Optional, Sequence, Tuple, Union, List
import math
import os
import zlib
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
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse
from torchmetrics import Metric
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.spatial.distance import cdist

# import math
from torch.utils.data.dataloader import default_collate
import pytorch_metric_learning.losses


from pyannote.audio import Inference
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import (
    Problem,
    Resolution,
    Specifications,
    get_dtype,
    Task,
)
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.audio.pipelines.clustering import (
    KMeansClustering,
    OracleClustering,
    AgglomerativeClustering,
)
from pyannote.audio.pipelines.utils import SpeakerDiarizationMixin

from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    SpeakerConfusionRate,
)

from torchmetrics import MetricCollection

# from pyannote.audio.utils.random import create_rng_for_worker


Subtask = Literal["diarization", "embedding"]

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)
Subtasks = list(Subtask.__args__)


def create_rng_for_worker(model) -> np.random.Generator:
    """Create worker-specific random number generator.

    Ensures reproducible training samples generation and unique seeds per worker and epoch.

    Parameters
    ----------
    model : YourModelType
        The model instance containing training state.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator with a unique seed.
    """

    global_seed = os.environ.get("PL_GLOBAL_SEED", "unset")
    worker_info = torch.utils.data.get_worker_info()

    worker_id = worker_info.id if worker_info else None

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
        duration: float = 10.0,
        max_speakers_per_chunk: int = 3,
        max_speakers_per_frame: int = 2,
        weigh_by_cardinality: bool = False,
        batch_size: int = 32,
        dia_task_rate: float = 0.5,
        num_workers: int = None,
        pin_memory: bool = False,
        margin: float = 11.4,
        small_margin: float = 5.7,
        scale: float = 30.0,
        alpha: float = 0.5,
        augmentation: BaseWaveformTransform = None,
        cache: Optional[Union[str, None]] = None,
        automatic_optimization: bool = True,
        diar_pooling: bool = False,
        keep_shorter_segments: bool = True,
        noise_augmentation: Optional[BaseWaveformTransform] = None,
        rir_augmentation: Optional[BaseWaveformTransform] = None,
        mean_var_norm: bool = False,
        gradient: dict = {
            "clip_val": 5.0,
            "clip_algorithm": "norm",
            "accumulate_batches": 1,
        },
        ami_aam_weight: float = 1.0,
        sample_utterances_from_same_file: bool = True,
        vc_dia_weight: float = 1.0,
        lambda_param: float = 0.8,
        normalize_utterances: bool = False,
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
        self.small_margin = small_margin
        self.scale = scale
        self.alpha = alpha
        # keep track of the use of database available in the meta protocol
        # * embedding databases are those with global speaker label scope
        # * diarization databases are those with file or database speaker label scope
        self.embedding_files_id = []
        self.global_files_id = []

        self.validation_metrics = MetricCollection(
            {
                "DiarizationErrorRate": DiarizationErrorRate(0.5),
                "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
                "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
                "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
            }
        )

        self.oracle_validation_metrics = self.validation_metrics.clone(prefix="Oracle")
        self.embedding_validation_metrics = MetricCollection(
            {
                "EqualErrorRate": EqualErrorRate(compute_on_cpu=True, distances=False),
                "BinaryAUROC": BinaryAUROC(compute_on_cpu=True),
            }
        )
        self.diar_pooling = diar_pooling
        self.keep_shorter_segments = keep_shorter_segments
        self.noise_augmentation = noise_augmentation
        self.rir_augmentation = rir_augmentation
        self.mean_var_norm = mean_var_norm
        self.gradient = gradient
        self.dia_task_rate = dia_task_rate
        self.ami_aam_weight = ami_aam_weight
        self.vc_dia_weight = vc_dia_weight
        self.sample_utterances_from_same_file = sample_utterances_from_same_file
        self.overlap = [0.0, 0.0]
        self.activation = [0.0, 0.0]
        self.total_dur = 0.0
        self.lambda_param = lambda_param
        self.normalize_utterances = normalize_utterances

    def val__getitem__(self, idx):
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
        # seems to be working kinda slow overall...
        # how does it work with annotated regions?
        file_idx = idx // self.batch_size
        chunk_idx = idx % self.batch_size

        file_id = self.prepared_data["validation-files-diar"][file_idx]
        validation_mask = self.prepared_data["audio-metadata"][
            "subset"
        ] == Subsets.index("development")
        diar_mask = self.prepared_data["audio-metadata"]["database"] != 0
        all_valid_files = np.argwhere(validation_mask & diar_mask).flatten()
        idx_in_val = np.where(all_valid_files == file_id)[0][0]
        file = next(
            itertools.islice(self.protocol.development(), idx_in_val, idx_in_val + 1)
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
        """Define length for the second validation dataset"""
        return len(self.prepared_data["validation-files-diar"]) * self.batch_size

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
                elif isinstance(value, Path):
                    metadatum[key] = str(value)
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
        diar_mask = prepared_data["audio-metadata"]["database"] != 0
        emb_mask = prepared_data["audio-metadata"]["database"] == 0
        # we dont want validation chunks to overlap too much
        length_mask = (
            prepared_data["audio-annotated"] >= self.batch_size * self.duration / 5
        )

        # Get all validation files that meet the criteria
        all_valid_files = np.argwhere(
            validation_mask & diar_mask & length_mask
        ).flatten()

        # Group files by database
        files_by_database = {}
        for file_idx in all_valid_files:
            database_idx = prepared_data["audio-metadata"]["database"][file_idx]
            if database_idx not in files_by_database:
                files_by_database[database_idx] = []
            files_by_database[database_idx].append(file_idx)

        # Randomly select at most 10 files from each database
        selected_files = []
        rng = np.random.default_rng(seed=42)  # For reproducibility
        for database_files in files_by_database.values():
            if len(database_files) > 10:
                selected_files.extend(
                    rng.choice(database_files, size=10, replace=False)
                )
            else:
                selected_files.extend(database_files)

        prepared_data["validation-files-diar"] = np.array(selected_files)

    def setup(self, stage="fit"):
        """Setup method

        Parameters
        ----------
        stage : {'fit', 'validate', 'test'}, optional
            Setup stage. Defaults to 'fit'.
        """

        super().setup()

        self.prepared_data["metadata-values"] = self.prepared_data[
            "metadata-values"
        ].tolist()

        self.global_files_id = np.argwhere(
            self.prepared_data["audio-metadata"]["scope"] > 1
        ).flatten()

        database_scope_mask = self.prepared_data["audio-metadata"]["scope"] > 0
        database_indexes = np.unique(
            self.prepared_data["audio-metadata"][database_scope_mask]["database"]
        )

        classes = {}
        for idx in database_indexes:
            database = self.prepared_data["metadata-values"]["database"][idx]
            # keep database training files
            database_files_id = np.argwhere(
                np.logical_and(
                    self.prepared_data["audio-metadata"]["database"] == idx,
                    self.prepared_data["audio-metadata"]["subset"]
                    == Subsets.index("train"),
                )
            ).flatten()
            database_mask = np.isin(
                self.prepared_data["annotations-segments"]["file_id"], database_files_id
            )
            max_idx = max(
                self.prepared_data["annotations-segments"][database_mask][
                    "database_label_idx"
                ]
            )
            classes[database] = np.arange(max_idx + 1)

        # if there is no file dedicated to the embedding task
        if self.alpha != 1.0 and len(classes) == 0:
            self.num_dia_samples = self.batch_size
            self.alpha = 1.0
            warnings.warn(
                "No class found for the speaker embedding task. Model will be trained on the speaker diarization task only."
            )

        if self.alpha != 0.0 and len(self.global_files_id) == len(
            self.prepared_data["audio-metadata"]
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
            classes=classes,
        )
        self.specifications = (speaker_diarization, speaker_embedding)

    def prepare_chunk(
        self, file_id: int, start_time: float, duration: float, database: str = None
    ):
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
        # there is a bug with arcface if we take global_label_idx
        if label_scope in ["database", "global"]:
            label_scope_key = "database_label_idx"
        else:
            label_scope_key = "file_label_idx"

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        mode = "pad" if self.keep_shorter_segments else "raise"
        sample["X"], _ = self.model.audio.crop(
            file, chunk, duration=duration, mode=mode
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
        y = np.zeros((int(num_frames), num_labels), dtype=np.uint8)

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
        # if sample["X"].shape[1] == 48000:
        #     print(sample["X"].shape)
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
        # TODO: check that chunk duration not longer than annotated duration
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

    def draw_embedding_utterance(
        self, rng: random.Random, class_id: int, duration: float, file_id=None
    ) -> tuple:
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
        if file_id is not None:
            class_segments_idx = (
                self.prepared_data["annotations-segments"]["file_id"] == file_id
            )
            class_segments = self.prepared_data["annotations-segments"][
                class_segments_idx
            ]
        else:
            class_segments_idx = (
                self.prepared_data["annotations-segments"]["global_label_idx"]
                == class_id
            )
            class_segments = self.prepared_data["annotations-segments"][
                class_segments_idx
            ]

        # sample one segment from all the class segments:
        segments_duration = class_segments["end"] - class_segments["start"]
        segments_total_duration = np.sum(segments_duration)
        # segment probability should be zero if duration less than duration but add to one
        prob_segments = segments_duration / segments_total_duration
        prob_segments[segments_duration < duration] = 0
        # Renormalize probabilities to sum to 1 after zeroing out invalid segments
        if np.sum(prob_segments) > 0:
            prob_segments = prob_segments / np.sum(prob_segments)
        else:
            # If no valid segments remain, raise an error
            raise ValueError(
                f"No valid segments of duration {duration}s or longer found for class {class_id}"
            )
        segment = rng.choice(class_segments, p=prob_segments)

        return (segment["file_id"], segment)

    def train__iter__helper(self, rng: random.Random, **filters):
        training_mask = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "train"
        )
        for key, value in filters.items():
            training_mask &= self.prepared_data["audio-metadata"][
                key
            ] == self.prepared_data["metadata"][key].index(value)
        file_ids = np.where(training_mask)[0]

        diar_mask = self.prepared_data["audio-metadata"]["database"] != 0
        diar_files_ids = np.where(diar_mask & training_mask)[0]
        embedding_files_ids = file_ids[np.isin(file_ids, self.global_files_id)]

        if self.num_dia_samples > 0:
            annotated_duration = self.prepared_data["audio-annotated"][diar_files_ids]
            cum_prob_annotated_duration = np.cumsum(
                annotated_duration / np.sum(annotated_duration)
            )

        duration = self.duration
        batch_size = self.batch_size

        emb_task_classes = deepcopy(
            self.specifications[Subtasks.index("embedding")].classes["VoxCeleb"]
        )
        rng.shuffle(emb_task_classes)
        sample_idx = 0
        embedding_class_idx = 0

        X_utterances: List[torch.Tensor] = []
        y_utterances: List[int] = []
        delays: List[float] = []
        utterance_durations: List[float] = []
        mixture_file_ids: List[int] = []
        klasses: List[str] = []
        files: List[int] = []

        while True:
            if sample_idx < self.num_dia_samples:
                file_id, start_time = self.draw_diarization_chunk(
                    diar_files_ids, cum_prob_annotated_duration, rng, duration
                )
                sample_diar = self.prepare_chunk(file_id, start_time, duration)
                sample_idx = (sample_idx + 1) % batch_size
                yield sample_diar
                continue

            if embedding_class_idx + self.max_speakers_per_chunk > len(emb_task_classes):
                rng.shuffle(emb_task_classes)
                embedding_class_idx = 0
            current_mixture_classes = emb_task_classes[
                embedding_class_idx : embedding_class_idx + self.max_speakers_per_chunk
            ]
            embedding_class_idx += self.max_speakers_per_chunk

            class_available_times = {klass: 0.0 for klass in current_mixture_classes}

            num_utterances_in_mixture = rng.integers(
                1, self.max_speakers_per_chunk + 1
            )

            X_utterances: List[torch.Tensor] = []
            delays: List[float] = []
            klasses: List[str] = []

            current_time = 0.0

            silence_in_the_end = rng.uniform(-9, 1.0)
            if silence_in_the_end < 0.0:
                silence_in_the_end = 0.0

            while True:
                if (
                    len(X_utterances) >= num_utterances_in_mixture
                    and delays[-1] + utterance_durations[-1] + silence_in_the_end
                    > self.duration
                ):
                    break

                available_classes = [
                    klass
                    for klass, avail_time in class_available_times.items()
                    if avail_time <= current_time
                ]

                if not available_classes:
                    next_available_time = min(class_available_times.values())
                    buffer_time = 0.5
                    current_time = next_available_time + buffer_time
                    available_classes = [
                        klass
                        for klass, avail_time in class_available_times.items()
                        if avail_time <= current_time
                    ]

                klass = rng.choice(available_classes)

                def truncated_exponential(rng, lambda_param, low, high):
                    F_low = 1 - np.exp(-lambda_param * low)
                    F_high = 1 - np.exp(-lambda_param * high)
                    u = rng.uniform(F_low, F_high)
                    return -np.log(1 - u) / lambda_param

                lambda_param = self.lambda_param
                low, high = 2.0, 10.0
                utterance_duration = truncated_exponential(rng, lambda_param, low, high)

                min_delay = current_time - class_available_times[klass]
                min_delay = max(min_delay, 0.0)
                delay = self.calculate_delay(
                    rng, utterance_durations, delays, extension=1.0, min_delay=min_delay
                )

                utterance_start_time = current_time + delay

                try:
                    file_id, speech_turn = self.draw_embedding_utterance(
                        rng,
                        klass,
                        utterance_duration,
                    )
                except ValueError:
                    continue

                delays.append(delay)
                utterance_durations.append(utterance_duration)
                mixture_file_ids.append(file_id)
                file = self.get_file(file_id)
                X = self.process_speech_turn(
                    file, rng, speech_turn, utterance_duration, delay
                )
                if self.normalize_utterances:
                    X = self.normalize_tensor(X)
                X_utterances.append(X)
                y_utterances.append(klass)
                klasses.append(klass)
                files.append(file_id)

                class_available_times[klass] = utterance_start_time + utterance_duration

            mixture = self.create_mixtures(X_utterances, delays, rng=rng)
            activation_features, unique_klasses = self.compute_activation_features(
                X_utterances, delays, klasses
            )

            num_frames = self.model.num_frames(
                round(duration * self.model.hparams.sample_rate)
            )

            y = (
                activation_features.transpose(0, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            labels = unique_klasses.copy()
            sample = {
                "X": mixture,
                "y": SlidingWindowFeature(y, self.model.receptive_field, labels=labels),
                "meta": {
                    "subset": 0,
                    "database": 0,
                    "file": mixture_file_ids[-1],
                    "scope": 2,
                },
            }

            num_utterances = 0
            X_utterances = []
            y_utterances = []
            delays = []
            utterance_durations = []
            mixture_file_ids = []
            sample_idx = (sample_idx + 1) % batch_size
            klasses = []
            files = []

            yield deepcopy(sample)

    def calculate_delay(
        self,
        rng: np.random.Generator,
        utterance_durations: List[float],
        delays: List[float],
        extension: float = 1.0,
        min_delay: float = 0.0,
    ) -> float:
        """Calculate the delay for the current utterance, allowing for extended silence.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        utterance_durations : List[float]
            List of utterance durations
        delays : List[float]
            List of previous delays
        extension : float, optional
            Additional maximum delay in seconds, by default 1.0

        Returns
        -------
        float
            Calculated delay
        """
        if not utterance_durations:
            # First utterance can be delayed by up to 1 second
            delay = rng.uniform(-9.0, 1.0)
            if delay < 0.0:
                delay = 0.0
        else:
            previous_delay = delays[-1]
            previous_duration = utterance_durations[-1]
            # Allow up to 1 second more than previous duration
            min_delay = max(min_delay, 0.5)
            delay = (
                rng.uniform(min_delay, previous_duration + extension) + previous_delay
            )
        return delay

    def select_speech_turn(
        self, rng: np.random.Generator, file: Dict, utterance_duration: float
    ) -> Segment:
        """Select a speech turn from the file, weighted by duration."""
        if not file["speech_turns"]:
            raise ValueError("No speech turns available in the selected file.")
        durations = [s.duration for s in file["speech_turns"]]
        total_duration = sum(durations)
        probabilities = [d / total_duration for d in durations]

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
        # speech turn is actually a segment
        duration = speech_turn["end"] - speech_turn["start"]
        if duration < utterance_duration:
            segment = Segment(speech_turn["start"], speech_turn["end"])
            X, _ = self.model.audio.crop(file, segment)
            num_missing_samples = (
                math.floor(utterance_duration * self.model.audio.sample_rate)
                - X.shape[1]
            )
            X = F.pad(X, (0, num_missing_samples))
        else:
            start_time = rng.uniform(
                speech_turn["start"], speech_turn["end"] - utterance_duration
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
        utterances: List[torch.Tensor],
        delays: List[float],
        rng: np.random.Generator,
        target_len: float = 10.0,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Create a mixture of audio utterances with random delays and gains."""
        target_len_samples = int(target_len * sample_rate)
        mixture = torch.zeros(1, target_len_samples)

        for utt, delay in zip(utterances, delays):
            start = int(delay * sample_rate)
            end = start + utt.shape[1]

            if end > target_len_samples:
                end = target_len_samples
                utt = utt[:, : end - start]
            if end - start < target_len_samples * 0.05:
                continue  # Skip if the utterance segment is too short

            gain_db = rng.uniform(0, 5)
            gain = self.db_to_amplitude(gain_db)
            mixture[:, start:end] += utt * gain

        if mixture.abs().sum() == 0:
            raise ValueError("Mixture is silent. Adjust delays or durations.")

        return mixture

    @staticmethod
    def db_to_amplitude(db: float) -> float:
        """Convert decibel value to amplitude."""
        return 10 ** (db / 20)

    def compute_activation_features(
        self,
        utterances: List[torch.Tensor],
        delays: List[float],
        klasses: List[str],
        target_len: float = 10.0,
        num_frames: int = 499,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Compute activation features for each speaker.

        Parameters
        ----------
        utterances : List[torch.Tensor]
            List of utterance tensors.
        delays : List[float]
            List of delays for each utterance.
        klasses : List[str]
            List of speaker classes corresponding to each utterance.
        target_len : float, optional
            Target length of the mixture in seconds, by default 10.0.
        num_frames : int, optional
            Number of frames for activation features, by default 499.
        sample_rate : int, optional
            Sampling rate in Hz, by default 16000.

        Returns
        -------
        torch.Tensor
            Activation features tensor of shape (num_speakers, 2, num_frames).
        """
        target_len_samples = int(target_len * sample_rate)
        unique_klasses = sorted(set(klasses))
        num_speakers = len(unique_klasses)

        binary_tensor = torch.zeros(num_speakers, 1, target_len_samples)

        klass_to_idx = {klass: idx for idx, klass in enumerate(unique_klasses)}

        for i, (utt, delay) in enumerate(zip(utterances, delays)):
            klass = klasses[i]
            speaker_idx = klass_to_idx[klass]
            start = int(delay * sample_rate)
            end = start + utt.shape[1]

            if end > target_len_samples:
                end = target_len_samples
                utt = utt[:, : end - start]

            if end - start < target_len_samples * 0.05:
                continue

            binary_tensor[speaker_idx, 0, start:end] = 1.0

        # # Compute "any other speaker active" channel
        # binary_tensor[:, 1, :] = (binary_tensor[:, 0, :].sum(dim=0, keepdim=True) > 0).float() - binary_tensor[:, 0, :]

        binary_tensor = binary_tensor.clamp(min=0.0)

        interp_signals = F.interpolate(
            binary_tensor, size=num_frames, mode="nearest"
        )  # Shape: (num_speakers, 2, num_frames)

        activation_features = interp_signals.squeeze(
            1
        )  # Shape: (num_speakers, num_frames)
        return activation_features, unique_klasses

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
        # seems to be working kinda slow overall...
        # how does it work with annotated regions?
        file_idx = idx // self.batch_size
        chunk_idx = idx % self.batch_size

        file_id = self.prepared_data["validation-files-diar"][file_idx]
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

    def collate_y_val(self, batch) -> torch.Tensor:
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

        for b in batch:
            # diarization reference
            y_dia = b["y"].data
            labels = b["y"].labels
            num_speakers = len(labels)
            # embedding reference

            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y_dia, axis=0), axis=0)
                # keep only the most talkative speakers
                y_dia = y_dia[:, indices[: self.max_speakers_per_chunk]]

            elif (
                num_speakers < self.max_speakers_per_chunk
                and y_dia.shape[1] < self.max_speakers_per_chunk
            ):
                # create inactive speakers by zero padding
                y_dia = np.pad(
                    y_dia,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            collated_y_dia.append(y_dia)

        return torch.from_numpy(np.stack(collated_y_dia))

    def collate_fn_val1(self, batch, stage="train"):
        collated = default_collate(batch)
        return collated

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
                if b["meta"]["scope"] > 0:
                    labels = np.array(labels)
                    y_emb = labels[indices[: self.max_speakers_per_chunk]]

            elif (
                num_speakers < self.max_speakers_per_chunk
                and y_dia.shape[1] < self.max_speakers_per_chunk
            ):
                # create inactive speakers by zero padding
                y_dia = np.pad(
                    y_dia,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )
                if b["meta"]["scope"] > 0:
                    y_emb[:num_speakers] = labels[:]

            else:
                if b["meta"]["scope"] > 0:
                    y_emb[:num_speakers] = labels[:]

            collated_y_dia.append(y_dia)
            collate_y_emb.append(y_emb)

        return (
            torch.from_numpy(np.stack(collated_y_dia)),
            torch.from_numpy(np.stack(collate_y_emb)),
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
        collated_X = self.collate_X(batch)

        collated_y_dia, collated_y_emb = self.collate_y(batch)

        collated_meta = self.collate_meta(batch)

        if self.noise_augmentation or self.rir_augmentation:
            # only augment VoxCeleb
            mask = collated_meta["database"] == 0

            if mask.any():
                samples_to_augment = collated_X[mask]

                if self.noise_augmentation:
                    self.noise_augmentation.train(mode=(stage == "train"))
                    samples_to_augment = self.noise_augmentation(
                        samples=samples_to_augment,
                        sample_rate=self.model.hparams.sample_rate,
                    )["samples"]

                collated_X[mask] = samples_to_augment

        collated_batch = {
            "X": collated_X,
            "y_dia": collated_y_dia,
            "y_emb": collated_y_emb,
            "meta": collated_meta,
        }

        # Include additional information for validation stage
        if stage == "val":
            collated_batch["annotation"] = batch[0]["annotation"]
            collated_batch["start_times"] = [b["start_time"] for b in batch]
            collated_batch["y"] = self.collate_y_val(batch)

        return collated_batch

    def setup_loss_func(self):

        classes = self.specifications[Subtasks.index("embedding")].classes
        # define a loss for each database-scope dataset
        arcface_losses = []
        available_databases = 0
        for database in classes:
            # we might have trained arc face weights on VoxCeleb
            if hasattr(self.model, "arc_face_loss") and hasattr(
                getattr(self.model, "arc_face_loss"), database
            ):
                available_databases += 1
            if database != "VoxCeleb":
                arcface_losses.append(
                    (
                        database,
                        ArcFaceLoss(
                            len(classes[database]),
                            self.model.hparams["embedding_dim"],
                            margin=self.small_margin,
                            scale=self.scale,
                            weight_init_func=nn.init.xavier_normal_,
                        ).to(self.model.device),
                    )
                )

            else:
                arcface_losses.append(
                    (
                        database,
                        ArcFaceLoss(
                            len(classes[database]),
                            self.model.hparams["embedding_dim"],
                            margin=self.margin,
                            scale=self.scale,
                            weight_init_func=nn.init.xavier_normal_,
                        ).to(self.model.device),
                    )
                )
        if available_databases < 2:
            self.model.arc_face_loss = nn.ModuleDict(
                {database: loss_func for database, loss_func in arcface_losses}
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

        dia_loss = self.segmentation_loss(prediction, permutated_target)
        return dia_loss

    def compute_embedding_loss(self, emb_prediction, target_emb, valid_embs, database):
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

        if len(emb_prediction.shape) == 3:
            embeddings = rearrange(emb_prediction, "b s e -> (b s) e")
        else:
            embeddings = emb_prediction
        targets = rearrange(target_emb, "b s -> (b s)")
        valid_embs = rearrange(valid_embs, "b s -> (b s)")

        emb_loss = torch.tensor(0)
        if self.diar_pooling:
            emb_loss = self.model.arc_face_loss[database](
                embeddings[valid_embs, :], targets[valid_embs]
            )
        else:
            raise ValueError("diar_pooling False not supported for multi-aam")
            max_pool = nn.MaxPool1d(kernel_size=3, stride=3)
            converted_targets = max_pool(targets.unsqueeze(0).float()).squeeze(0).long()
            converted_valid_embs = (
                max_pool(valid_embs.unsqueeze(0).float()).squeeze(0).bool()
            )
            emb_loss = self.model.arc_face_loss[database](
                embeddings[converted_valid_embs, :],
                converted_targets[converted_valid_embs],
            )

        if torch.isnan(emb_loss):
            return None

        self.model.log(
            f"loss/train/arcface/{database}",
            emb_loss,
            on_step=True,
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

        waveform = batch["X"]
        target_dia = batch["y_dia"]
        target_emb = batch["y_emb"]
        # drop samples that contain too many speakers
        num_speakers = torch.sum(torch.any(target_dia, dim=1), dim=1)
        keep = num_speakers <= self.max_speakers_per_chunk

        target_dia = target_dia[keep]
        target_emb = target_emb[keep]
        waveform = waveform[keep]

        num_remaining_dia_samples = torch.sum(keep[: self.num_dia_samples])
        batch_size = keep.shape[0]
        # corner case
        if not keep.any():
            return None

        # forward pass
        dia_prediction, emb_prediction = self.model(waveform)

        # get the best permutation
        dia_multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target_dia, permut_map = permutate(dia_multilabel, target_dia)
        permutated_target_emb = target_emb[
            torch.arange(target_emb.shape[0]).unsqueeze(1), permut_map
        ]

        agree = permutated_target_dia == dia_multilabel
        agree_pos = torch.logical_and(dia_multilabel > 0, agree)
        valid_embs = torch.sum(agree_pos, dim=1) > 0.1 * dia_multilabel.shape[1]

        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target_dia.float()
        )

        dia_prediction = dia_prediction
        permutated_target_powerset = permutated_target_powerset

        dia_loss = torch.tensor(0)

        # if batch contains diarization subtask chunks, then compute diarization loss on these chunks:
        if self.alpha != 0.0 and self.dia_task_rate != 0.0:
            dia_loss_vc = self.compute_diarization_loss(
                dia_prediction[num_remaining_dia_samples:],
                permutated_target_powerset[num_remaining_dia_samples:],
            )
            dia_loss_ami = self.compute_diarization_loss(
                dia_prediction[:num_remaining_dia_samples],
                permutated_target_powerset[:num_remaining_dia_samples],
            )
            self.model.log(
                "loss/train/dia_voxceleb",
                dia_loss_vc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.model.log(
                "loss/train/dia_ami",
                dia_loss_ami,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            dia_loss = (
                num_remaining_dia_samples * dia_loss_ami
                + (batch_size - num_remaining_dia_samples)
                * dia_loss_vc
                * self.vc_dia_weight
            )
            dia_loss = dia_loss / batch_size
        # to model device
        emb_loss = torch.tensor(0).to(self.model.device)
        # if batch contains embedding subtask chunks, then compute embedding loss on these chunks:
        if self.alpha != 1.0 and torch.any(valid_embs):

            if self.dia_task_rate != 1.0:
                emb_database = "VoxCeleb"
                emb_loss = (
                    self.compute_embedding_loss(
                        emb_prediction[num_remaining_dia_samples:],
                        permutated_target_emb[num_remaining_dia_samples:],
                        valid_embs[num_remaining_dia_samples:],
                        emb_database,
                    )
                    * (batch_size - num_remaining_dia_samples)
                    / batch_size
                )

                emb_database_idx = batch["meta"]["database"][0]
                emb_database = self.prepared_data["metadata-values"]["database"][
                    emb_database_idx
                ]
                emb_loss_ami = None

                if emb_loss_ami is not None:
                    emb_loss += (
                        self.ami_aam_weight
                        * emb_loss_ami
                        * num_remaining_dia_samples
                        / batch_size
                    )

            else:
                emb_database_idx = batch["meta"]["database"][0]
                emb_database = self.prepared_data["metadata-values"]["database"][
                    emb_database_idx
                ]
                emb_loss = self.compute_embedding_loss(
                    emb_prediction[:num_remaining_dia_samples],
                    permutated_target_emb[:num_remaining_dia_samples],
                    valid_embs[:num_remaining_dia_samples],
                    emb_database,
                )
            loss = self.alpha * dia_loss + (1 - self.alpha) * emb_loss
        else:
            loss = self.alpha * dia_loss

        # using multiple optimizers requires manual optimization
        if not self.model.automatic_optimization:
            optimizers = self.model.optimizers()
            optimizers = optimizers if isinstance(optimizers, list) else [optimizers]

            num_accumulate_batches = self.gradient["accumulate_batches"]
            if batch_idx % num_accumulate_batches == 0:
                for optimizer in optimizers:
                    optimizer.zero_grad()

            scaled_loss = loss / num_accumulate_batches

            self.model.manual_backward(scaled_loss)

            if (batch_idx + 1) % num_accumulate_batches == 0:
                for optimizer in optimizers:
                    self.model.clip_gradients(
                        optimizer,
                        # gradient_clip_val=90,
                        # gradient_clip_algorithm="autoclip",
                        gradient_clip_val=5.0,
                        gradient_clip_algorithm="norm",
                    )
                    optimizer.step()

            if self.model.lr_schedulers() is not None:
                sch = self.model.lr_schedulers()
                if isinstance(sch, list):
                    for s in sch:
                        s.step()
                else:
                    sch.step()

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

        # gradient is not needed here, so we can detach tensors from the gradient graph
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

        # let's also calculate standard DER
        target = batch["y"]
        multilabel = self.model.powerset.to_multilabel(segmentations)

        self.model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target, 1, 2),
        )
        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return None

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components for diarization subtask,
        and equal error rate for the embedding part
        """
        return {
            "LocalDiarizationErrorRate": DiarizationErrorRate(0.5),
            "LocalDiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
            "LocalDiarizationErrorRate/Miss": MissedDetectionRate(0.5),
            "LocalDiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
        }

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor to have zero mean and unit variance."""
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        return (tensor - mean) / (std + 1e-8)  # Add epsilon to prevent division by zero
