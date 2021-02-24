"""
reference : Qiu, J., Tang, J., Ma, H., Dong, Y., Wang, K., Tang, J., 2018.
Deepinf:  Social769influence  prediction  with  deep  learning, in:
Proceedings  of  the  24th  ACM770SIGKDD International Conference on Knowledge Discovery & Data Mining,771pp. 2110–2119.
https://github.com/xptree/DeepInf
"""

#!/usr/bin/env python
# encoding: utf-8
# File Name: train.py
# Author: Jiezhong Qiu
# Create Time: 2017/11/08 07:43
# TODO:

# Modified for the paper "Exploring the representational power of Graph Autoencoder"
# Added gnn_mean and gnn_sum

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tensorboard_logger import tensorboard_logger
from torch.utils.data import DataLoader

from data_loader import ChunkSampler
from data_loader import InfluenceDataSet
from data_loader import PatchySanDataSet
from gat import BatchGAT
from gcn import BatchGCN
from gnn_sum import BatchGNNSUM
from gnn_mean import BatchGNNMEAN
from pscn import BatchPSCN


def evaluate(model, args, class_weight, logger, epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        output = model(features, vertices, graph, args.model_type)
        if args.model == "gcn" or args.model == "gat" or args.model == "gnn_sum"or args.model == "gnn_mean":
            output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                log_desc, loss / total, auc, prec, rec, f1)

    tensorboard_logger.log_value(log_desc + 'loss', loss / total, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'prec', prec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'rec', rec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def train(model, args, class_weight, logger, optimizer, epoch, train_loader, valid_loader, test_loader,
          log_desc='train_'):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        optimizer.zero_grad()
        output = model(features, vertices, graph, args.model_type)
        if args.model == "gcn" or args.model == "gat" or args.model == "gnn_sum"or args.model == "gnn_mean":
            output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(model, args, class_weight, logger,epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(model, args, class_weight, logger,epoch, test_loader, thr=best_thr, log_desc='test_')


# Train model
def train_evaluate_model(data_path, model_name, model_type, hidden_units, instance_normalization, class_weight_balanced,
                         use_vertex_feature):
    print("EXPERIMENT START------",model_name,model_type,hidden_units)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-log', type=str, default='', help="name of this run")
    parser.add_argument('--model', type=str, default=model_name, help="models used")
    parser.add_argument('--model_type', type=str, default=model_type, help="model type used")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--hidden-units', type=str, default=hidden_units,
                        help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument('--heads', type=str, default="1,1,1",
                        help="Heads in each layer, splitted with comma")
    parser.add_argument('--batch', type=int, default=1024, help="Batch size")
    parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--check-point', type=int, default=10, help="Eheck point")
    parser.add_argument('--instance-normalization', action='store_true', default=instance_normalization,
                        help="Enable instance normalization")
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    parser.add_argument('--file-dir', type=str, required=False, default=data_path, help="Input file directory")
    parser.add_argument('--train-ratio', type=float, default=75, help="Training ratio (0, 100)")
    parser.add_argument('--valid-ratio', type=float, default=12.5, help="Validation ratio (0, 100)")
    parser.add_argument('--class-weight-balanced', action='store_true', default=class_weight_balanced,
                        help="Adjust weights inversely proportional"
                             " to class frequencies in the input data")
    parser.add_argument('--use-vertex-feature', action='store_true', default=use_vertex_feature,
                        help="Whether to use vertices' structural features")
    parser.add_argument('--sequence-size', type=int, default=16,
                        help="Sequence size (only useful for pscn)")
    parser.add_argument('--neighbor-size', type=int, default=5,
                        help="Neighborhood size (only useful for pscn)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    tensorboard_log_dir = 'tensorboard/%s_%s' % (args.model, args.tensorboard_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    tensorboard_logger.configure(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

    # adj N*n*n
    # feature N*n*f
    # labels N*n*c
    # Load data
    # vertex: vertex id in global network N*n

    if args.model == "pscn":
        influence_dataset = PatchySanDataSet(
            args.file_dir, args.dim, args.seed, args.shuffle, args.model,
            sequence_size=args.sequence_size, stride=1, neighbor_size=args.neighbor_size)
    else:
        influence_dataset = InfluenceDataSet(
            args.file_dir, args.dim, args.seed, args.shuffle, args.model)

    N = len(influence_dataset)
    n_classes = 2
    class_weight = influence_dataset.get_class_weight() \
        if args.class_weight_balanced else torch.ones(n_classes)
    logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])

    feature_dim = influence_dataset.get_feature_dimension()
    n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]
    logger.info("feature dimension=%d", feature_dim)
    logger.info("number of classes=%d", n_classes)

    train_start, valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
    train_loader = DataLoader(influence_dataset, batch_size=args.batch,
                              sampler=ChunkSampler(valid_start - train_start, 0))
    valid_loader = DataLoader(influence_dataset, batch_size=args.batch,
                              sampler=ChunkSampler(test_start - valid_start, valid_start))
    test_loader = DataLoader(influence_dataset, batch_size=args.batch,
                             sampler=ChunkSampler(N - test_start, test_start))

    # Model and optimizer
    if args.model == "gcn":
        model = BatchGCN(pretrained_emb=influence_dataset.get_embedding(),
                         vertex_feature=influence_dataset.get_vertex_features(),
                         use_vertex_feature=args.use_vertex_feature,
                         n_units=n_units,
                         dropout=args.dropout,
                         instance_normalization=args.instance_normalization)
    elif args.model == "gnn_sum":
        model = BatchGNNSUM(pretrained_emb=influence_dataset.get_embedding(),
                            vertex_feature=influence_dataset.get_vertex_features(),
                            use_vertex_feature=args.use_vertex_feature,
                            n_units=n_units,
                            dropout=args.dropout,
                            instance_normalization=args.instance_normalization,
                            model_type=args.model_type)
    elif args.model == "gnn_mean":
        model = BatchGNNMEAN(pretrained_emb=influence_dataset.get_embedding(),
                            vertex_feature=influence_dataset.get_vertex_features(),
                            use_vertex_feature=args.use_vertex_feature,
                            n_units=n_units,
                            dropout=args.dropout,
                            instance_normalization=args.instance_normalization,
                            model_type=args.model_type)
    elif args.model == "gat":
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        model = BatchGAT(pretrained_emb=influence_dataset.get_embedding(),
                         vertex_feature=influence_dataset.get_vertex_features(),
                         use_vertex_feature=args.use_vertex_feature,
                         n_units=n_units, n_heads=n_heads,
                         dropout=args.dropout, instance_normalization=args.instance_normalization)
    elif args.model == "pscn":
        model = BatchPSCN(pretrained_emb=influence_dataset.get_embedding(),
                          vertex_feature=influence_dataset.get_vertex_features(),
                          use_vertex_feature=args.use_vertex_feature,
                          n_units=n_units,
                          dropout=args.dropout,
                          instance_normalization=args.instance_normalization,
                          sequence_size=args.sequence_size,
                          neighbor_size=args.neighbor_size)
    else:
        raise NotImplementedError

    if args.cuda:
        model.cuda()
        class_weight = class_weight.cuda()

    params = [{'params': filter(lambda p: p.requires_grad, model.parameters())
    if args.model == "pscn" else model.layer_stack.parameters()}]

    optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    logger.info("training...")
    for epoch in range(args.epochs):
        train(model, args, class_weight, logger, optimizer,epoch, train_loader, valid_loader, test_loader)
    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    logger.info("retrieve best threshold...")
    best_thr = evaluate(model, args, class_weight, logger,args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')

    # Testing
    logger.info("testing...")
    evaluate(model, args, class_weight, logger,args.epochs, test_loader, thr=best_thr, log_desc='test_')

    print("EXPERIMENT END------",model_name,model_type,hidden_units)
