"""
Display a scatter plot of the embeddings colored according to the node labels and the
topological features bin labels

"""

# load libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import pretreatment.utils as ut

def visualise_single_embedding(embedding, model_name, plot_index, axs):
    """
    Visualize the embeddings of a model
    :param embedding: the embedding matrix
    :param plot_index: the index of the sub plot (column)
    :param axs: axs object of the subplot
    :return:
    """

    # reduce dimensionality with TSNE to 2 dimension
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embedding)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    area = np.pi * 3
    # dot_colors = list(mcolors.CSS4_COLORS)
    # random.shuffle(dot_colors)
    dot_colors = ["blue", "red", "orange", "green", "yellow", "cyan", "purple", "black", "pink"]

    label_types = ["node_labels", "degree_lables", "betweenness_centrality_lables", "clustering_lables",
                   "eigenvector_centrality_lables","triangles_lables"]
    titles = [model_name + " " + "label", model_name + " " + "degree", model_name + " " + "betweenness",
              model_name + " " + "clustering", model_name + " " + "eigenvector", model_name + " " + "triangles"]

    plot_secondary_index = 0
    # loop through the label types and plot it
    for label_type in label_types:
        if label_type == "node_labels":
            labels = ut.node_labels
        else:
            labels = ut.load_numpy_file(ut.topo_features_labels_path + label_type + ".npy")
        number_classes = len(set(labels))

        xc = []
        yc = []
        for c in range(0, number_classes):
            xc.append([])
            yc.append([])
            for i in range(0, len(ut.graph.nodes)):
                if labels[i] == c:
                    xc[c].append(x[i])
                    yc[c].append(y[i])
            axs[plot_secondary_index, plot_index].scatter(xc[c], yc[c], s=area, c=dot_colors[c], alpha=0.5)
            axs[plot_secondary_index, plot_index].set_title(titles[plot_secondary_index])
            axs[plot_secondary_index, plot_index].axis('off')
        plot_secondary_index += 1


def visualize_results(save_plot= False):
    """
    visualize the embeddings of multiple models in a scatter plot
    """
    fig, axs = plt.subplots(6, 10)
    fig.set_figheight(16)
    fig.set_figwidth(16)
    embedding = ut.load_numpy_file(ut.embedding_path + "gae_first_embedding.npy")
    visualise_single_embedding(embedding,"gae_first", 0, axs)
    embedding = ut.load_numpy_file(ut.embedding_path + "gae_concat_embedding.npy")
    visualise_single_embedding(embedding,"gae_concat", 1, axs)
    embedding = ut.load_numpy_file(ut.embedding_path + "gae_mixed_embedding.npy")
    visualise_single_embedding(embedding,"gae_mixed", 2, axs)
    embedding = ut.load_numpy_file(ut.embedding_path + "gae_l1_sum_embedding.npy")
    visualise_single_embedding(embedding,"gae_l1_sum", 3, axs)
    embedding = ut.load_numpy_file(ut.embedding_path + "matrix_factorization_embedding.npy")
    visualise_single_embedding(embedding,"MF", 4, axs)

    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    if save_plot :
        plt.savefig('embedding.png', dpi=200)
    plt.show()


def visualize_Large(model_name, feature):
    """
    visualize one large plot for a model and
    color it according to certain feature labels
    :param model_name:
    :param feature:
    """
    embedding = ut.load_numpy_file(ut.embedding_path + model_name + "_embedding.npy")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embedding)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    area = np.pi * 3
    dot_colors = ["blue", "red", "orange", "green", "yellow", "cyan", "purple", "black", "pink"]

    label_types = [feature + "_lables"]
    plot_secondary_index = 0
    for label_type in label_types:
        if label_type == "node_labels":
            labels = ut.node_labels
        else:
            labels = ut.load_numpy_file(ut.topo_features_labels_path + label_type + ".npy")
        number_classes = len(set(labels))

        xc = []
        yc = []
        for c in range(0, number_classes):
            xc.append([])
            yc.append([])
            for i in range(0, len(ut.graph.nodes)):
                if labels[i] == c:
                    xc[c].append(x[i])
                    yc[c].append(y[i])
            plt.scatter(xc[c], yc[c], s=area, c=dot_colors[c], alpha=0.5)
        plot_secondary_index += 1
    plt.show()
