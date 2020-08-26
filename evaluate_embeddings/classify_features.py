"""
This file contains the list of functions that classify the topolofical features
using the embeddings.
the following classification models are used:
 -Logistic regression
 -SVM linear
 -SVM RBF
 = MPL (2 layers)
"""

# import libraries
import numpy as np
import math

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabulate import tabulate

import pretreatment.utils as ut

dic_score = {}

def classify_features(embedding_model_name, feature_name):
    """
    classify the topological features using the model embeddings
    :param embedding_model_name: the model that generated the embeddings
    :param feature_name: the topo feature that we are classifying
    """
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_embedding.npy")
    y = ut.load_numpy_file(ut.topo_features_labels_path + feature_name + "_lables.npy")

    max_iter = 200

    # linear regression
    r = LogisticRegression(max_iter=max_iter)
    scores = cross_validate(r, x, y, scoring=["neg_mean_squared_error", "neg_mean_absolute_error",])
    add_score(embedding_model_name, feature_name, 'r_mse', abs(np.mean(scores['test_neg_mean_squared_error'])))
    add_score(embedding_model_name, feature_name, 'r_mae', abs(np.mean(scores['test_neg_mean_absolute_error'])))

    # logistic regression
    lr = LogisticRegression(max_iter=max_iter)
    scores = cross_validate(lr, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, feature_name, 'lr_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, feature_name, 'lr_f1_micro', np.mean(scores['test_f1_micro']))

    # SVM linear
    svm = SVC(kernel='linear', C=1, max_iter=max_iter)
    scores = cross_validate(svm, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, feature_name, 'svm_ln_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, feature_name, 'svm_ln_f1_micro', np.mean(scores['test_f1_micro']))

    # SVM Kernel RBF
    svm = SVC(kernel='rbf', C=1, max_iter=max_iter)
    scores = cross_validate(svm, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, feature_name, 'svm_rbf_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, feature_name, 'svm_rbf_f1_micro', np.mean(scores['test_f1_micro']))

    # MPL 2 Layers
    mlp = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=max_iter)
    scores = cross_validate(mlp, x, y, scoring=["f1_macro", "f1_micro"])
    add_score(embedding_model_name, feature_name, 'mlp_f1_macro', np.mean(scores['test_f1_macro']))
    add_score(embedding_model_name, feature_name, 'mlp_f1_micro', np.mean(scores['test_f1_micro']))


def add_score(embedding_model_name, feature_name, score_name, value):
    global dic_score
    dic_score[embedding_model_name][feature_name][score_name].append(round(value, 3))


def setup_score(embedding_model_name, feature_name):
    global dic_score
    if not embedding_model_name in dic_score:
        dic_score[embedding_model_name] = {}
    if not feature_name in dic_score[embedding_model_name]:
        dic_score[embedding_model_name][feature_name] = {}

    dic_score[embedding_model_name][feature_name]["r_mse"] = []
    dic_score[embedding_model_name][feature_name]["r_mae"] = []
    dic_score[embedding_model_name][feature_name]["lr_f1_macro"] = []
    dic_score[embedding_model_name][feature_name]["lr_f1_micro"] = []
    dic_score[embedding_model_name][feature_name]["svm_ln_f1_macro"] = []
    dic_score[embedding_model_name][feature_name]["svm_ln_f1_micro"] = []
    dic_score[embedding_model_name][feature_name]["svm_rbf_f1_macro"] = []
    dic_score[embedding_model_name][feature_name]["svm_rbf_f1_micro"] = []
    dic_score[embedding_model_name][feature_name]["mlp_f1_macro"] = []
    dic_score[embedding_model_name][feature_name]["mlp_f1_micro"] = []


def print_score():
    print("======================")
    print("CLASSIFYING FEATURES")
    print("======================")

    for emebdding_model in dic_score:
        all_scores = []
        print("..............")
        print(emebdding_model)
        print("..............")
        for feature in dic_score[emebdding_model]:
            scores = (feature,)
            for score in dic_score[emebdding_model][feature]:
                scores = scores + (str(np.round(np.mean(dic_score[emebdding_model][feature][score]),3)),)
            all_scores.append(scores)

        print(tabulate(all_scores,
                       headers=["feature","r_mse","r_mae", "lr_f1_macro", "lr_f1_micro", "svm_ln_f1_macro", "svm_ln_f1_micro",
                                "svm_rbf_f1_macro",
                                "svm_rbf_f1_micro", "mlp_f1_macro", "mlp_f1_micro"]))
