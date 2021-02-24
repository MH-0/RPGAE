"""
Classify the node labels (Ground truth labels of the graph)
using the embeddings of each embedding module
the classification is done with:
logistic regression
svm linear
svm rbf
mlp 2 layers

"""

# import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabulate import tabulate

import pretreatment.utils as ut

dic_score = {}


def classify_embeddings(embedding_model_name):
    """
    classify the embeddings of each embedding model
    :param embedding_model_name: the name of the embedding model
    :return:
    """

    # load embedding
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_embedding.npy")
    # load classes
    y = ut.node_labels
    max_iter = 200

    # logistic regression
    lr = LogisticRegression(max_iter=max_iter)
    scores = cross_validate(lr, x, y, scoring=["f1_macro", "f1_micro"])

    add_score(embedding_model_name, 'lr_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, 'lr_f1_micro', np.mean(scores['test_f1_micro']))

    # svm linear
    svm = SVC(kernel='linear', C=1, max_iter=max_iter)
    scores = cross_validate(svm, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, 'svm_ln_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, 'svm_ln_f1_micro', np.mean(scores['test_f1_micro']))

    # svm rbf
    svm = SVC(kernel='rbf', C=1, max_iter=max_iter)
    scores = cross_validate(svm, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, 'svm_rbf_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, 'svm_rbf_f1_micro', np.mean(scores['test_f1_micro']))

    # mlp
    mlp = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=max_iter)
    scores = cross_validate(mlp, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, 'mlp_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, 'mlp_f1_micro', np.mean(scores['test_f1_micro']))


def add_score(embedding_model_name, score_name, value):
    global dic_score
    dic_score[embedding_model_name][score_name].append(round(value, 3))


def setup_score(embedding_model_name):
    """
    setup score dictionary to be filled
    :param embedding_model_name: the name of the embedding module
    :return:
    """
    global dic_score
    if not embedding_model_name in dic_score:
        dic_score[embedding_model_name] = {}

    dic_score[embedding_model_name]["lr_f1_macro"] = []
    dic_score[embedding_model_name]["lr_f1_micro"] = []
    dic_score[embedding_model_name]["svm_ln_f1_macro"] = []
    dic_score[embedding_model_name]["svm_ln_f1_micro"] = []
    dic_score[embedding_model_name]["svm_rbf_f1_macro"] = []
    dic_score[embedding_model_name]["svm_rbf_f1_micro"] = []
    dic_score[embedding_model_name]["mlp_f1_macro"] = []
    dic_score[embedding_model_name]["mlp_f1_micro"] = []


def save_score():
    """
    Save the score of the classification
    """
    np.save(ut.data_path + "scores\\classify_embeddings_score", dic_score)
