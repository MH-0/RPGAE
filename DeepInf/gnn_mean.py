# Added for the paper "Exploring the representational power of Graph Autoencoder"
"""
This file contains the definition of gnn_mean
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_mean_layers import BatchGraphMEAN


class BatchGNNMEAN(nn.Module):
    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature,
                 use_vertex_feature, fine_tune=False, instance_normalization=False,model_type="concat"):
        super(BatchGNNMEAN, self).__init__()
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
            l1 = n_units[i]
            l2 = n_units[i + 1]
            self.layer_stack.append(
                BatchGraphMEAN(l1, l2)
            )

    def forward(self, x, vertices, lap,model_type):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap,i,model_type)
            if i + 1 < self.num_layer:
                x = torch.tanh(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
