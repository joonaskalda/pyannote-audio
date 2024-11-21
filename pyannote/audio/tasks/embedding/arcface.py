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


from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import pytorch_metric_learning.losses
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from .mixins import SupervisedRepresentationLearningTaskMixin

#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class AAMSoftmax(AbsLoss):
    """Additive angular margin softmax.

    Paper: Deng, Jiankang, et al. "Arcface: Additive angular margin loss for
    deep face recognition." Proceedings of the IEEE/CVF conference on computer
    vision and pattern recognition. 2019.

    args:
        nout    : dimensionality of speaker embedding
        nclases: number of speakers in the training set
        margin  : margin value of AAMSoftmax
        scale   : scale value of AAMSoftmax
    """

    def __init__(
        self, nout, nclasses, margin=0.3, scale=15, easy_margin=False, **kwargs
    ):
        super().__init__(nout)

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nout
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print("Initialised AAMSoftmax margin %.3f scale %.3f" % (self.m, self.s))

    def forward(self, x, label=None):
        if len(label.size()) == 2:
            label = label.squeeze(1)

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        return loss


class ArcMarginProduct_intertopk_subcenter(AbsLoss):
    r"""Implement of large margin arc distance with intertopk and subcenter:

    Reference:
        MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
        FOR SPEAKER VERIFICATION.
        https://arxiv.org/pdf/2110.05042.pdf
        Sub-center ArcFace: Boosting Face Recognition by
        Large-Scale Noisy Web Faces.
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta + margin)
        K: number of sub-centers
        k_top: number of hard samples
        mp: margin penalty of hard samples
        do_lm: whether do large margin finetune
    """

    def __init__(
        self,
        nout,
        nclasses,
        scale=30.0,
        margin=0.3,
        easy_margin=False,
        K=3,
        mp=0.06,
        k_top=5,
        do_lm=False,
    ):
        super().__init__(nout)
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin
        )  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

        self.ce = nn.CrossEntropyLoss()

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        if len(label.size()) == 2:
            label = label.squeeze(1)
        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K)
        )  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi)
                + (top_k_one_hot * phi_mp)
                + ((1.0 - one_hot - top_k_one_hot) * cosine)
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.ce(output, label)
        return loss


class SupervisedRepresentationLearningWithArcFace(
    SupervisedRepresentationLearningTaskMixin,
):
    """Supervised representation learning with ArcFace loss

    Representation learning is the task of ...

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

    """

    #  TODO: add a ".metric" property that tells how speaker embedding trained with this approach
    #  should be compared. could be a string like "cosine" or "euclidean" or a pdist/cdist-like
    #  callable. this ".metric" property should be propagated all the way to Inference (via the model).

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
        sampling_mode: str = "classes_weighted_file_uniform",
    ):

        self.num_chunks_per_class = num_chunks_per_class
        self.num_classes_per_batch = num_classes_per_batch

        self.margin = margin
        self.scale = scale

        self.drop_last = False
        self.seed = 42
        
        self.sampling_mode = sampling_mode

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

    def setup_loss_func(self):
        self.model.eval()
        _, embedding_size = self.model(self.model.example_input_array).shape
        self.model.train()
        self.model.loss_func = pytorch_metric_learning.losses.ArcFaceLoss(
            len(self.specifications.classes),
            embedding_size,
            margin=self.margin,
            scale=self.scale,
            weight_init_func=nn.init.xavier_normal_
        )
        # self.model.loss_func = AAMSoftmax(
        #     embedding_size,
        #     len(self.specifications.classes),
        #     margin=0.3,
        #     scale=30,
        # )