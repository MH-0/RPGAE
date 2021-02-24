"""
This file contains the list of functions that classify the topolofical features
using the embeddings.
the following classification models are used:
 -Logistic regression
 -SVM linear
 -SVM RBF
 - MPL (2 layers)

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
    scores = cross_validate(r, x, y, scoring=["neg_mean_squared_error", "neg_mean_absolute_error", ])
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
    dic_score[feature_name][embedding_model_name][score_name].append(round(value, 3))


def setup_score(embedding_model_name, feature_name):
    global dic_score
    if not feature_name in dic_score:
        dic_score[feature_name] = {}
    if not embedding_model_name in dic_score[feature_name]:
        dic_score[feature_name][embedding_model_name] = {}

    dic_score[feature_name][embedding_model_name]["r_mse"] = []
    dic_score[feature_name][embedding_model_name]["r_mae"] = []
    dic_score[feature_name][embedding_model_name]["lr_f1_macro"] = []
    dic_score[feature_name][embedding_model_name]["lr_f1_micro"] = []
    dic_score[feature_name][embedding_model_name]["svm_ln_f1_macro"] = []
    dic_score[feature_name][embedding_model_name]["svm_ln_f1_micro"] = []
    dic_score[feature_name][embedding_model_name]["svm_rbf_f1_macro"] = []
    dic_score[feature_name][embedding_model_name]["svm_rbf_f1_micro"] = []
    dic_score[feature_name][embedding_model_name]["mlp_f1_macro"] = []
    dic_score[feature_name][embedding_model_name]["mlp_f1_micro"] = []


def save_score():

    best_models = {}
    best_model_per_score = {}

    # determine best model per feature per score
    for feature in dic_score:
        best_model_per_score[feature] = {}
        best_models[feature] = {}
        for emebdding_model in dic_score[feature]:
            best_models[feature][emebdding_model] = 0
            for score in dic_score[feature][emebdding_model]:
                score_mean_value = np.mean(dic_score[feature][emebdding_model][score])
                if not score in best_model_per_score[feature]:
                    best_model_per_score[feature][score] = emebdding_model
                else:
                    # for mse and mae we take the lowest
                    if score in ["r_mse","r_mae"]:
                        if score_mean_value <  np.mean(dic_score[feature][best_model_per_score[feature][score]][score]):
                            best_model_per_score[feature][score] = emebdding_model
                    # for the rest of scores we take the highest
                    else:
                        if score_mean_value >  np.mean(dic_score[feature][best_model_per_score[feature][score]][score]):
                            best_model_per_score[feature][score] = emebdding_model

    # for each model count the number of times it is has best performance
    for feature in best_models:
        print(feature)
        for model in best_models[feature]:
            for score in best_model_per_score[feature]:
                if best_model_per_score[feature][score] == model:
                    best_models[feature][model]+=1
            print(model,best_models[feature][model])

    # save score
    np.save(ut.data_path + "scores\\classify_features_score", dic_score)
    np.save(ut.data_path + "scores\\best_models", best_models)
