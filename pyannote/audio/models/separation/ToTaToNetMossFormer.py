# MIT License
#
# Copyright (c) 2020 CNRS
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)
from asteroid_filterbanks import make_enc_dec
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.masknn import DPRNN
from asteroid.models.one_path_flash_fsmn import SBFLASHBlock_DualA,Dual_Path_Model_MossFormer
from transformers import AutoProcessor, AutoModel

class ToTaToNetMossFormer(Model):
    """ToTaToNet joint speaker diarization and speech separation model

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    diar : dict, optional
        Keyword arguments used to initalize the average pooling in the diarization branch.
        Defaults to {"frames_per_second": 125}.
    encoder_decoder : dict, optional
        Keyword arguments used to initalize the encoder and decoder.
        Defaults to {"fb_name": "free", "kernel_size": 32, "n_filters": 64, "stride": 16}.
    dprnn : dict, optional
        Keyword arguments used to initalize the DPRNN model.
        Defaults to {"n_repeats": 6, "bn_chan": 128, "hid_size": 128, "chunk_size": 100, "norm_type": "gLN", "mask_act": "relu", "rnn_type": "LSTM"}.
    sample_rate : int, optional
        Audio sample rate. Defaults to 16000.
    num_channels : int, optional
        Number of channels. Defaults to 1.
    task : Task, optional
        Task to perform. Defaults to None.
    n_sources : int, optional
        Number of sources. Defaults to 3.
    use_lstm : bool, optional
        Whether to use LSTM in the diarization branch. Defaults to False.
    use_wavlm : bool, optional
        Whether to use the WavLM large model for feature extraction. Defaults to True.
    gradient_clip_val : float, optional
        Gradient clipping value. Required when fine-tuning the WavLM model and thus using two different optimizers. 
        Defaults to 5.0.
    """

    ENCODER_DECODER_DEFAULTS = {
        "fb_name": "free",
        "kernel_size": 32,
        "n_filters": 512,
        "stride": 16,
    }
    LSTM_DEFAULTS = {
        "hidden_size": 64,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 64, "num_layers": 0}
    MOSSFORMER2_DEFAULTS = {
        "intra_numlayers": 8,
        "bn_chan": 512,
        "intra_nhead":8,
        "intra_dffn":1024,
        "dropout":0,
        "intra_use_positional": True,
        "intra_norm_before": True,
        "num_layers":1,
        "norm_type": "ln",
        "chunk_size":250,
        "masknet_extraskipconnection": True,
        "masknet_useextralinearlayer": False
    }
    DIAR_DEFAULTS = {"frames_per_second": 125}

    def __init__(
        self,
        encoder_decoder: dict = None,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        diar: Optional[dict] = None,
        convnet: dict = None,
        mossformer2: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        n_sources: int = 6,
        use_lstm: bool = False,
        use_wavlm: bool = True,
        gradient_clip_val: float = 5.0,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        mossformer2 = merge_dict(self.MOSSFORMER2_DEFAULTS, mossformer2)
        encoder_decoder = merge_dict(self.ENCODER_DECODER_DEFAULTS, encoder_decoder)
        diar = merge_dict(self.DIAR_DEFAULTS, diar)
        self.n_src = n_sources
        self.use_lstm = use_lstm
        self.use_wavlm = use_wavlm
        self.save_hyperparameters("encoder_decoder", "lstm", "linear", "mossformer2", "diar")
        self.n_sources = n_sources

        if encoder_decoder["fb_name"] == "free":
            n_feats_out = encoder_decoder["n_filters"]
        elif encoder_decoder["fb_name"] == "stft":
            n_feats_out = int(2 * (encoder_decoder["n_filters"] / 2 + 1))
        else:
            raise ValueError("Filterbank type not recognized.")
        self.encoder, self.decoder = make_enc_dec(
            sample_rate=sample_rate, **self.hparams.encoder_decoder
        )
    

        #---- Setting up the Intra MossFormer block -----#
        intra_model = SBFLASHBlock_DualA(
            num_layers=mossformer2['intra_numlayers'],
            d_model=mossformer2['bn_chan'],
            nhead=mossformer2['intra_nhead'],
            d_ffn=mossformer2['intra_dffn'],
            dropout=mossformer2['dropout'],
            use_positional_encoding=mossformer2['intra_use_positional'],
            norm_before=mossformer2['intra_norm_before'],
        )

        if self.use_wavlm:
            self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-large")
            downsampling_factor = 1
            for conv_layer in self.wavlm.feature_extractor.conv_layers:
                if isinstance(conv_layer.conv, nn.Conv1d):
                    downsampling_factor *= conv_layer.conv.stride[0]
            self.wavlm_scaling = int(downsampling_factor / encoder_decoder["stride"])

            self.masker = Dual_Path_Model_MossFormer(
                in_channels=encoder_decoder["n_filters"] + self.wavlm.feature_projection.projection.out_features,
                btln_channels=mossformer2['bn_chan'],
                out_channels=encoder_decoder["n_filters"],
                intra_model=intra_model,
                num_layers=mossformer2["num_layers"],
                norm=mossformer2["norm_type"],
                K=mossformer2["chunk_size"],
                num_spks=n_sources,
                skip_around_intra=mossformer2["masknet_extraskipconnection"],
                linear_layer_after_inter_intra=mossformer2["masknet_useextralinearlayer"]
            )
        else:
            self.masker = DPRNN(
                encoder_decoder["n_filters"],
                out_chan=encoder_decoder["n_filters"],
                n_src=n_sources,
                **self.hparams.dprnn
            )

        # diarization can use a lower resolution than separation
        self.diarization_scaling = int(
            sample_rate / diar["frames_per_second"] / encoder_decoder["stride"]
        )
        self.average_pool = nn.AvgPool1d(
            self.diarization_scaling, stride=self.diarization_scaling
        )
        linaer_input_features = n_feats_out
        if self.use_lstm:
            del lstm["monolithic"]
            multi_layer_lstm = dict(lstm)
            self.lstm = nn.LSTM(n_feats_out, **multi_layer_lstm)
            linaer_input_features = lstm["hidden_size"] * (
                2 if lstm["bidirectional"] else 1
            )
        if linear["num_layers"] > 0:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [
                            linaer_input_features,
                        ]
                        + [self.hparams.linear["hidden_size"]]
                        * self.hparams.linear["num_layers"]
                    )
                ]
            )
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return 1

    def build(self):
        if self.use_lstm or self.hparams.linear["num_layers"] > 0:
            self.classifier = nn.Linear(64, self.dimension)
        else:
            self.classifier = nn.Linear(1, self.dimension)
        self.activation = self.default_activation()

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

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_num_frames(
            num_samples, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_receptive_field_size(
            num_frames, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_receptive_field_center(
            frame, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        sources : (batch, sample, n_sources)
        """
        bsz = waveforms.shape[0]
        tf_rep = self.encoder(waveforms)
        if self.use_wavlm:
            wavlm_rep = self.wavlm(waveforms.squeeze(1)).last_hidden_state
            wavlm_rep = wavlm_rep.transpose(1, 2)
            wavlm_rep = wavlm_rep.repeat_interleave(self.wavlm_scaling, dim=-1)
            wavlm_rep = pad_x_to_y(wavlm_rep, tf_rep)
            wavlm_rep = torch.cat((tf_rep, wavlm_rep), dim=1)
            masks = self.masker(wavlm_rep)
        else:
            masks = self.masker(tf_rep)
        # shape: (batch, nsrc, nfilters, nframes)
        masked_tf_rep = masks * tf_rep.unsqueeze(1)
        decoded_sources = self.decoder(masked_tf_rep)
        decoded_sources = pad_x_to_y(decoded_sources, waveforms)
        decoded_sources = decoded_sources.transpose(1, 2)
        outputs = torch.flatten(masked_tf_rep, start_dim=0, end_dim=1)
        # shape (batch * nsrc, nfilters, nframes)
        outputs = self.average_pool(outputs)
        outputs = outputs.transpose(1, 2)
        # shape (batch, nframes, nfilters)
        if self.use_lstm:
            outputs, _ = self.lstm(outputs)
        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
        if not self.use_lstm and self.hparams.linear["num_layers"] == 0:
            outputs = (outputs**2).sum(dim=2).unsqueeze(-1)
        outputs = self.classifier(outputs)
        outputs = outputs.reshape(bsz, self.n_sources, -1)
        outputs = outputs.transpose(1, 2)

        return self.activation[0](outputs), decoded_sources
