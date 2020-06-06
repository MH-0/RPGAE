"""
This file analyses the topological features (handcrafted) and divide them into classes (via binning)
in order to be classified later with the embeddings
"""

# load libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
from scipy.stats import entropy

import pretreatment.utils as ut


def calculate_topo_features():
    """
    Calculate topological features
    """
    print(dict(nx.degree(ut.graph)))
    print(nx.degree_centrality(ut.graph))
    print(nx.triangles(ut.graph))
    print(nx.clustering(ut.graph))
    print(nx.eigenvector_centrality_numpy(ut.graph))
    print(nx.katz_centrality_numpy(ut.graph))
    print(nx.pagerank_numpy(ut.graph))
    print(nx.betweenness_centrality(ut.graph))


def plot_fit_power_law(values, title, index):
    """
    This function plots the distributions
    And fits it to power law distribution to verify
    if the data follows a power law
    :param values: list of values
    :param title: title of the plot
    :param index: index of the plot (should be incremental by 1)
    """

    # first line subplot
    a = 240 + index
    plt.subplot(a, title=title)
    sns.distplot(values, rug=True)

    # second line subplot
    a = a + 4
    plt.subplot(a)
    fit = powerlaw.Fit(np.array(values))
    fit.power_law.plot_pdf(color='b', linestyle='--', label='fit ccdf')
    fit.plot_pdf(color='b')
    print('alpha= ', fit.power_law.alpha, '  sigma= ', fit.power_law.sigma)


def verify_feature_power_law():
    """
    Verifies that the topological features
    follow a power law distribution
    """
    # Total Degree
    plot_fit_power_law(list(dict(nx.degree(ut.graph)).values()), "Degree", 1)
    # Local clustering
    plot_fit_power_law(list(nx.clustering(ut.graph).values()), "Local clustering", 2)
    # eigenvector centrality
    plot_fit_power_law(list(nx.eigenvector_centrality_numpy(ut.graph).values()), "Eigenvector centrality", 3)
    # betweenness centrality
    plot_fit_power_law(list(nx.betweenness_centrality(ut.graph).values()), "Betweenness centrality", 4)
    plt.show()


def save_topo_features():
    """
    concatenate topological features into one array
    and save them to a file
    """
    degree = np.array(list(dict(nx.degree(ut.graph)).values()))
    degree_centrality = np.array(list(nx.degree_centrality(ut.graph).values()))
    # triangles = np.array(list(nx.triangles(ut.graph).values()))
    triangles = degree
    clustering = np.array(list(nx.clustering(ut.graph).values()))
    eigenvector_centrality = np.array(list(nx.eigenvector_centrality_numpy(ut.graph).values()))
    katz_centrality = np.array(list(nx.katz_centrality_numpy(ut.graph).values()))
    pagerank = np.array(list(nx.pagerank_numpy(ut.graph).values()))
    betweenness_centrality = np.array(list(nx.betweenness_centrality(ut.graph).values()))


    np.save(ut.topo_features_path + "degree.npy", degree)
    np.save(ut.topo_features_path + "degree_centrality.npy", degree_centrality)
    np.save(ut.topo_features_path + "triangles.npy", triangles)
    np.save(ut.topo_features_path + "clustering.npy", clustering)
    np.save(ut.topo_features_path + "eigenvector_centrality.npy", eigenvector_centrality)
    np.save(ut.topo_features_path + "katz_centrality.npy", katz_centrality)
    np.save(ut.topo_features_path + "pagerank.npy", pagerank)
    np.save(ut.topo_features_path + "betweenness_centrality.npy", betweenness_centrality)

    # concatenate the features (in case to be used as entry)
    degree = np.reshape(degree, (-1, 1))
    degree_centrality = np.reshape(degree_centrality, (-1, 1))
    triangles = np.reshape(triangles, (-1, 1))
    clustering = np.reshape(clustering, (-1, 1))
    eigenvector_centrality = np.reshape(eigenvector_centrality, (-1, 1))
    katz_centrality = np.reshape(katz_centrality, (-1, 1))
    pagerank = np.reshape(pagerank, (-1, 1))
    betweenness_centrality = np.reshape(betweenness_centrality, (-1, 1))

    topofeatures = np.concatenate([degree, degree_centrality], axis=1)
    topofeatures = np.concatenate([topofeatures, triangles], axis=1)
    topofeatures = np.concatenate([topofeatures, clustering], axis=1)
    topofeatures = np.concatenate([topofeatures, eigenvector_centrality], axis=1)
    topofeatures = np.concatenate([topofeatures, katz_centrality], axis=1)
    topofeatures = np.concatenate([topofeatures, pagerank], axis=1)
    topofeatures = np.concatenate([topofeatures, betweenness_centrality], axis=1)
    np.save(ut.topo_features_path + "topofeatures.npy", topofeatures)
    topofeatures = ut.load_numpy_file(ut.topo_features_path + "topofeatures.npy")

    print("Toplogical Features Calculated:")
    print(topofeatures)
    print("-------------")


def bin_feature(feature, feature_name, with_log, bin_count):
    """
    This function splits the topo feature values into bins
    and calculates the entropy of the labels in order
    to verify the balance of the classes
    :param feature: numpy array of values
    :param feature_name: name of the topological feature
    :param with_log: apply log to values
    :param bin_count: number of bins to split the values
    :return: labels of the features (to which bin they belong)
    """

    print(feature_name, ":")
    print("-----------")
    if with_log:
        feature = np.log(feature)

    qc = pd.qcut(feature, q=bin_count, precision=1,duplicates='drop')
    bins = qc.categories
    print("bins:", bins)
    codes = qc.codes

    print("labels:", codes)
    (unique, counts) = np.unique(codes, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    print("frequencies:\n", frequencies)
    print("Entropy:", entropy(counts))
    print("")

    # # show bins
    # df_count = qc.value_counts()
    # df_count.plot.bar(rot=0,title="eumails: degree bins")
    # plt.show()

    # save the labels to a file path for each feature
    np.save(ut.topo_features_labels_path + feature_name + "_lables.npy", codes)
    return codes


def generate_features_labels(bins):
    """
    this function divides all the features into classes (bins)
    :param bins: The number of bins
    """
    degree = ut.load_numpy_file(ut.topo_features_path + "degree.npy")
    degree_centrality = ut.load_numpy_file(ut.topo_features_path + "degree_centrality.npy")
    triangles = ut.load_numpy_file(ut.topo_features_path + "triangles.npy")
    clustering = ut.load_numpy_file(ut.topo_features_path + "clustering.npy")
    eigenvector_centrality = ut.load_numpy_file(ut.topo_features_path + "eigenvector_centrality.npy")
    katz_centrality = ut.load_numpy_file(ut.topo_features_path + "katz_centrality.npy")
    pagerank = ut.load_numpy_file(ut.topo_features_path + "pagerank.npy")
    betweenness_centrality = ut.load_numpy_file(ut.topo_features_path + "betweenness_centrality.npy")

    bin_feature(degree, "degree", False, bins)
    bin_feature(degree_centrality, "degree_centrality", False, bins)
    bin_feature(triangles, "triangles", False, bins)
    bin_feature(clustering, "clustering", False, bins)
    bin_feature(eigenvector_centrality, "eigenvector_centrality", False, bins)
    bin_feature(katz_centrality, "katz_centrality", False, bins)
    bin_feature(pagerank, "pagerank", False, bins)
    bin_feature(betweenness_centrality, "betweenness_centrality", False, bins)