"""
reference : Qiu, J., Tang, J., Ma, H., Dong, Y., Wang, K., Tang, J., 2018.
Deepinf:  Social769influence  prediction  with  deep  learning, in:
Proceedings  of  the  24th  ACM770SIGKDD International Conference on Knowledge Discovery & Data Mining,771pp. 2110–2119.
https://github.com/xptree/DeepInf
"""

#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/17 14:11
# TODO:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn_layers import BatchGraphConvolution


class BatchGCN(nn.Module):
    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature,
                 use_vertex_feature, fine_tune=False, instance_normalization=False):
        super(BatchGCN, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        self.layer_stack = nn.ModuleList()

        for i in range(self.num_layer):
            self.layer_stack.append(
                BatchGraphConvolution(n_units[i], n_units[i + 1])
            )

    def forward(self, x, vertices, lap, model_type):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
