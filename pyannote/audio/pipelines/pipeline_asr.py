# pipeline_asr.py

import functools
import itertools
import math
import textwrap
import warnings
from typing import Callable, Optional, Text, Tuple, Union, Dict

import numpy as np
from scipy.ndimage import binary_dilation
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.metrics.asr import CpWER

from pyannote.pipeline.parameter import Categorical, ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.pipelines.utils.diarization import set_num_speakers
from pyannote.audio.utils.signal import binarize

from whisperx import load_model as load_whisperx_model
from whisper.normalizers import EnglishTextNormalizer

from typing import List

from pyannote.audio.pipelines.speech_separation import SpeechSeparation


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeechSeparationASR(SpeechSeparation):
    """Speech separation and ASR pipeline

    Extends the SpeechSeparation pipeline to include ASR predictions for each speaker.

    Parameters
    ----------
    All parameters from SpeechSeparation, plus:
    asr_model_version : str
        Version of the WhisperX ASR model to use.
    asr_options : dict, optional
        Additional options for the WhisperX ASR model.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/separation-ami-1.0",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
        asr_model_version: str = "small.en",  # Default WhisperX model
        asr_options: Optional[dict] = None,
    ):
        super().__init__(
            segmentation=segmentation,
            segmentation_step=segmentation_step,
            embedding=embedding,
            embedding_exclude_overlap=embedding_exclude_overlap,
            clustering=clustering,
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            der_variant=der_variant,
            use_auth_token=use_auth_token,
        )

        # Initialize ASR model
        self.asr_model_version = asr_model_version
        self.asr_options = asr_options or {
            "max_new_tokens": None,
            "clip_timestamps": None,
            "hallucination_silence_threshold": None,
            "hotwords": None,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisperx_model = load_whisperx_model(
            self.asr_model_version,
            device_string,
            compute_type=compute_type,
            asr_options=self.asr_options,
        )
        self.normalizer = EnglishTextNormalizer()

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"

    def apply_asr(self, separated_audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Apply ASR to a separated audio source.

        Parameters
        ----------
        separated_audio : np.ndarray
            Audio waveform of the separated source.
        sample_rate : int, optional
            Sample rate of the audio. Defaults to 16000.

        Returns
        -------
        transcription : str
            ASR transcription of the audio.
        """
        if separated_audio.size == 0:
            return ""

        # Normalize audio
        audio = np.float32(separated_audio / np.max(np.abs(separated_audio) + 1e-8))

        # Transcribe using WhisperX
        try:
            result = self.whisperx_model.transcribe(
                audio, batch_size=16, language="en" #, sample_rate=sample_rate
            )
            text = " ".join([segment["text"] for segment in result["segments"]])
            normalized_text = self.normalizer(text)
            return normalized_text
        except Exception as e:
            warnings.warn(f"ASR transcription failed: {e}")
            return ""

    def apply(
        self,
        file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        hook: Optional[Callable] = None,
    ) -> Tuple[Annotation, Dict[str, str], Optional[np.ndarray]]:
        """Apply speaker diarization and ASR

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        return_embeddings : bool, optional
            Return representative speaker embeddings.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        asr_predictions : Dict[str, str]
            ASR transcriptions for each speaker.
        embeddings : np.array, optional
            Representative speaker embeddings such that `embeddings[i]` is the
            speaker embedding for i-th speaker in diarization.labels().
            Only returned when `return_embeddings` is True.
        """

        # Call the parent apply method to get diarization, sources, and optionally embeddings
        try:
            result = super().apply(
                file,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_embeddings,
                hook=hook,
            )
        except Exception as e:
            warnings.warn(f"Super apply failed: {e}")
            return Annotation(uri=file["uri"]), {}, None

        if return_embeddings:
            if len(result) != 3:
                warnings.warn(
                    "Expected 3 outputs from super().apply when return_embeddings=True"
                )
                diarization, sources = result[:2]
                embeddings = None
            else:
                diarization, sources, embeddings = result
        else:
            if len(result) != 2:
                warnings.warn(
                    "Expected 2 outputs from super().apply when return_embeddings=False"
                )
                diarization, sources, _ = result
            else:
                diarization, sources = result
                embeddings = None

        # Initialize ASR predictions dictionary
        # asr_predictions: Dict[str, str] = {}

        # If no speakers detected, return empty ASR predictions
        if not diarization:
            return diarization, asr_predictions, embeddings

        asr_predictions = [self.apply_asr(source) for source in sources.data.T]
        asr_predictions = [self.normalizer(prediction) for prediction in asr_predictions]
        # # Iterate over each speaker and perform ASR
        # for speaker in diarization.labels():
        #     # Get segments for the current speaker
        #     speaker_segments = diarization.get_timeline(label=speaker)

        #     # Initialize an empty list to hold audio chunks for the speaker
        #     speaker_audio = np.array([], dtype=np.float32)

        #     # Concatenate all segments for the speaker
        #     for segment in speaker_segments:
        #         # Extract audio for the segment
        #         waveform, sr = self._audio.crop(file, segment)
        #         if sr != self.whisperx_model.sample_rate:
        #             # Resample if necessary
        #             waveform = self._audio.resample(waveform, sr, self.whisperx_model.sample_rate)
        #         speaker_audio = np.concatenate((speaker_audio, waveform.flatten()))

        #     # Apply ASR to the concatenated audio
        #     transcription = self.apply_asr(
        #         speaker_audio, sample_rate=self.whisperx_model.sample_rate
        #     )
        #     transcription = self.apply_asr(source)
        #     # Store the transcription
        #     asr_predictions[speaker] = transcription

        return diarization, asr_predictions, sources
    
    def get_metric(self) -> CpWER:
        return CpWER(**self.der_variant)