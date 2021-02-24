"""
This file contains the list of functions used to evaluate
the homogeneity of the clusters

"""

import numpy as np
import sklearn.metrics as sk
from tabulate import tabulate

import pretreatment.utils as ut

dic_score = {}


def calculate_similarity(embedding_model_name):
    """
    calculate the similarity score of the clusters
        -davies_bouldin_score
        -calinski_harabasz
        -silhouette_score
    :param embedding_model_name:
    :return:
    """
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_embedding.npy")
    y = ut.node_labels

    davies_bouldin = sk.davies_bouldin_score(x, y)
    calinski_harabasz = sk.calinski_harabasz_score(x, y)
    silhouette_score = sk.silhouette_score(x, y)

    add_score(embedding_model_name, 'davies_bouldin', davies_bouldin)
    add_score(embedding_model_name, 'calinski_harabasz', calinski_harabasz)
    add_score(embedding_model_name, 'silhouette_score', silhouette_score)


def add_score(embedding_model_name, score_name, value):
    global dic_score
    dic_score[embedding_model_name][score_name].append(round(value, 3))


def setup_score(embedding_model_name):
    global dic_score
    if not embedding_model_name in dic_score:
        dic_score[embedding_model_name] = {}

    dic_score[embedding_model_name]["davies_bouldin"] = []
    dic_score[embedding_model_name]["calinski_harabasz"] = []
    dic_score[embedding_model_name]["silhouette_score"] = []


def save_score():
    """
    Save the score of the similarity
    """
    np.save(ut.data_path + "scores\\cluser_similarity_score", dic_score)
