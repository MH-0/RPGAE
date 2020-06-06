"""
Link prediction of edges
the attributes are the element wise product of the embeddings
of the node pairs for positive and negative edges.
The negative edges are sampled randomly.

The prediction is done with:
logistic regression
svm linear
svm rbf
mlp 2 layers
"""

import networkx as nx
# import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabulate import tabulate

import pretreatment.utils as ut

dic_score = {}


def prepare_link_prediction_data(embedding_model_name):
    # load embedding
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_embedding.npy")
    positive_samples = []
    negative_samples = []
    classes = []

    nodes_list = np.array(list(ut.graph.nodes()))

    for edge in ut.graph.edges:
        node1_index = np.where(nodes_list == edge[0])[0][0]
        node2_index = np.where(nodes_list == edge[1])[0][0]
        positive_samples.append(np.multiply(x[node1_index], x[node2_index]))
        classes.append(1)

    non_edges = list(nx.non_edges(ut.graph))
    non_edges = np.asarray(non_edges)
    sample_num = len(positive_samples)
    non_edge_samples = non_edges[np.random.choice(len(non_edges), sample_num, replace=False)]
    for edge in non_edge_samples:
        node1_index = np.where(nodes_list == edge[0])[0][0]
        node2_index = np.where(nodes_list == edge[1])[0][0]
        negative_samples.append(np.multiply(x[node1_index], x[node2_index]))
        classes.append(0)

    link_prediction_data = positive_samples + negative_samples
    np.save(ut.embedding_path + embedding_model_name + "_linkpredictiondata.npy", link_prediction_data)
    np.save(ut.embedding_path + embedding_model_name + "_linkpredictionclasses.npy", classes)


def predict_links(embedding_model_name):
    """
    classify the embeddings of each embedding model
    :param embedding_model_name: the name of the embedding model
    :return:
    """

    # load embedding
    x = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_linkpredictiondata.npy")
    # load classes
    y = ut.load_numpy_file(ut.embedding_path + embedding_model_name + "_linkpredictionclasses.npy")
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


def print_score():
    """
    Print the score of the classification
    """
    all_scores = []

    print("")
    print("======================")
    print("LINK PREDICTION")
    print("======================")

    for emebdding_model in dic_score:
        scores = (emebdding_model,)
        for score in dic_score[emebdding_model]:
            scores = scores + (str(np.round(np.mean(dic_score[emebdding_model][score]), 3)),)
        all_scores.append(scores)

    print(tabulate(all_scores,
                   headers=["model", "lr_f1_macro", "lr_f1_micro", "svm_ln_f1_macro", "svm_ln_f1_micro",
                            "svm_rbf_f1_macro",
                            "svm_rbf_f1_micro", "mlp_f1_macro", "mlp_f1_micro"]))
