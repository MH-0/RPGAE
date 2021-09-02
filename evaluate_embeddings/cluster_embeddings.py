"""
This file contains the list of functions that clusters the embeddings
using k-means and Finch
"""

# import libraries
import numpy as np
import scipy.optimize as op
import sklearn.metrics as sk
from sklearn.cluster import KMeans
from tabulate import tabulate

import pretreatment.utils as ut
from evaluate_embeddings.finch import FINCH

# holds the results of the  clustering task
dic_score = {}


def score_clustering_accuracy(ground_truth, predicted):
    """
    This function calculates the clustering accuracy
    :param ground_truth: the real clusters
    :param predicted: the predicted clusters
    :return:
    """
    y_predicted = np.array(predicted)
    y_true = np.array(ground_truth).astype(np.int64)
    D = max(y_predicted.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_predicted.size):
        w[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = op.linear_sum_assignment(-w)
    acc = w[row_ind, col_ind].sum() / y_predicted.size
    return acc


def cluster_embeddings(embedding_model_name):
    """
    Cluster the embeddings of a model using K-means
    :param embedding_model_name: the name of the model that generated the embeddings
    """
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_embedding.npy")
    y = ut.node_labels

    clusters = KMeans(n_clusters=ut.number_classes).fit(x)
    predicted = clusters.labels_

    arindex = sk.adjusted_rand_score(y, predicted)
    clustering_accuracy = score_clustering_accuracy(y, predicted)
    nmi = sk.normalized_mutual_info_score(y, predicted)

    add_score(embedding_model_name, 'kmeans-acc', clustering_accuracy)
    add_score(embedding_model_name, 'kmeans-nmi', nmi)
    add_score(embedding_model_name, 'kmeans-ari', arindex)

    c, num_clust, predicted = FINCH(x, req_clust=7,verbose=False)

    arindex = sk.adjusted_rand_score(y, predicted)
    clustering_accuracy = score_clustering_accuracy(y, predicted)
    nmi = sk.normalized_mutual_info_score(y, predicted)

    add_score(embedding_model_name, 'finch-acc', clustering_accuracy)
    add_score(embedding_model_name, 'finch-nmi', nmi)
    add_score(embedding_model_name, 'finch-ari', arindex)


def add_score(embedding_model_name, score_name, value):
    """
    Adds the score
    fills the variable dic_score according to the embedding model and the metric
    (Rounds the result to 3 decimal points)
    :param embedding_model_name: The name of the embedding model
    :param score_name: The metric name
    :param value: the results value
    """
    global dic_score
    dic_score[embedding_model_name][score_name].append(round(value, 3))


def setup_score(embedding_model_name):
    """
    Prepares the score dictionary to be filled
    Initiates the dic_score variable with the metrics and the embedding model name, to be filled later
    with the results
    :param embedding_model_name: the name of the embedding module
    """
    global dic_score
    if not embedding_model_name in dic_score:
        dic_score[embedding_model_name] = {}

    dic_score[embedding_model_name]["kmeans-acc"] = []
    dic_score[embedding_model_name]["kmeans-nmi"] = []
    dic_score[embedding_model_name]["kmeans-ari"] = []

    dic_score[embedding_model_name]["finch-acc"] = []
    dic_score[embedding_model_name]["finch-nmi"] = []
    dic_score[embedding_model_name]["finch-ari"] = []


def save_score():
    """
    Save the score of the clustering
    """
    np.save(ut.data_path + "scores\\cluster_embeddings_score", dic_score)
