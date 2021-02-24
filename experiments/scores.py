"""
This file contains the list of functions used to calculate and display the scores of the experiments
"""

# load libraries
import numpy as np
from tabulate import tabulate


def print_classify_features_score(dataset_name):
    """
    This function prints the Topological features classification experiments score
    :param dataset_name: the name of the dataset of which we want to print the experiments scores
    """

    print("======================")
    print("CLASSIFYING FEATURES")
    print("======================")

    dic_score = np.load("data\\" + dataset_name + "\\scores\\classify_features_score.npy", allow_pickle=True).item()

    for feature in dic_score:
        all_scores = []
        print("..............")
        print(feature)
        print("..............")

        for emebdding_model in dic_score[feature]:
            scores = (emebdding_model,)
            for score in dic_score[feature][emebdding_model]:
                score_mean_value = np.mean(dic_score[feature][emebdding_model][score])
                score_mean = str(np.round(score_mean_value, 3))
                score_std = str(np.round(np.std(dic_score[feature][emebdding_model][score]), 3))
                score_full = score_mean + " " + u"\u00B1" + score_std
                scores = scores + (score_full,)

            all_scores.append(scores)

        print(tabulate(all_scores,
                       headers=["model", "r_mse", "r_mae", "lr_f1_macro", "lr_f1_micro", "svm_ln_f1_macro",
                                "svm_ln_f1_micro",
                                "svm_rbf_f1_macro",
                                "svm_rbf_f1_micro", "mlp_f1_macro", "mlp_f1_micro"]))


def print_classify_embeddings_score(dataset_name):
    """
    This function prints the label classification experiments score
    :param dataset_name: the name of the dataset of which we want to print the experiments scores
    """
    all_scores = []

    print("")
    print("======================")
    print("CLASSIFYING NODE LABELS")
    print("======================")

    dic_score = np.load("data\\" + dataset_name + "\\scores\\classify_embeddings_score.npy", allow_pickle=True).item()

    for emebdding_model in dic_score:
        scores = (emebdding_model,)
        for score in dic_score[emebdding_model]:
            score_mean = str(np.round(np.mean(dic_score[emebdding_model][score]), 3))
            score_std = str(np.round(np.std(dic_score[emebdding_model][score]), 3))
            score_full = score_mean + " " + u"\u00B1" + score_std
            scores = scores + (score_full,)
        all_scores.append(scores)

    print(tabulate(all_scores,
                   headers=["model", "lr_f1_macro", "lr_f1_micro", "svm_ln_f1_macro", "svm_ln_f1_micro",
                            "svm_rbf_f1_macro",
                            "svm_rbf_f1_micro", "mlp_f1_macro", "mlp_f1_micro"]))


def print_cluster_embeddings_score(dataset_name):
    """
    This function prints the label clustering experiments score
    :param dataset_name: the name of the dataset of which we want to print the experiments scores
    """
    all_scores = []
    print("")
    print("======================")
    print("CLUSTER NODE LABELS")
    print("======================")

    dic_score = np.load("data\\" + dataset_name + "\\scores\\cluster_embeddings_score.npy", allow_pickle=True).item()

    for emebdding_model in dic_score:
        scores = (emebdding_model,)
        for score in dic_score[emebdding_model]:
            score_mean = str(np.round(np.mean(dic_score[emebdding_model][score]), 3))
            score_std = str(np.round(np.std(dic_score[emebdding_model][score]), 3))
            score_full = score_mean + " " + u"\u00B1" + score_std
            scores = scores + (score_full,)
        all_scores.append(scores)

    print(tabulate(all_scores,
                   headers=["model", "kmeans-acc", "kmeans-nmi", "kmeans-ari", "finch-acc", "finch-nmi", "finch-ari"]))


def print_cluser_similarity_score(dataset_name):
    """
    This function prints the homogeneity experiments score
    :param dataset_name: the name of the dataset of which we want to print the experiments scores
    """
    all_scores = []
    print("")
    print("======================")
    print("CLUSTERS SIMILARITY")
    print("======================")

    dic_score = np.load("data\\" + dataset_name + "\\scores\\cluser_similarity_score.npy", allow_pickle=True).item()

    for emebdding_model in dic_score:
        scores = (emebdding_model,)
        for score in dic_score[emebdding_model]:
            score_mean = str(np.round(np.mean(dic_score[emebdding_model][score]), 3))
            score_std = str(np.round(np.std(dic_score[emebdding_model][score]), 3))
            score_full = score_mean + " " + u"\u00B1" + score_std
            scores = scores + (score_full,)
        all_scores.append(scores)

    print(tabulate(all_scores,
                   headers=["model", "davies_bouldin", "calinski_harabasz", "silhouette_score"]))


def print_scores(dataset_name):
    """
    This function prints all experiements scores for a dataset
    :param dataset_name: the name of the dataset of which we want to print the experiments scores
    """
    print("\n\n")
    print("**********************************************************************")
    print("DATASET: ",dataset_name)
    print("**********************************************************************")

    # print topological features classification scores
    print_classify_features_score(dataset_name)
    # print labels classification scores
    print_classify_embeddings_score(dataset_name)
    # print labels clustering scores
    print_cluster_embeddings_score(dataset_name)
    # print homogeneity scores
    print_cluser_similarity_score(dataset_name)
