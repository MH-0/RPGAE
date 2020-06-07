"""
Generate the embeddings of each model
Graph auto-enders
Matrix Factorization
Node2vec
"""

# import libraries
import warnings

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import spectral_embedding

import embedding_models.gae as gae
import embedding_models.gae_mixed as gam
import pretreatment.utils as ut

warnings.filterwarnings("ignore")


def gae_embedding(model_name):
    """
    generate the embeddings of the Graph autoenders
    :param model_name: name of the gae model
    (gae_l1_sum,gae_l2_sum, gae_mean, gae_concat,gae_first,gae_spectral,gae_mixed)
    :return: embedding
    """

    if model_name == "gae_concat":
        layer_size = 32
    else:
        layer_size = 64

    # train model
    if model_name == "gae_mixed":
        embedding = gam.train(ut.graph, ut.input, ut.input_size, layer_size, layer_size,
                              250,
                              10, False)
    else:
        embedding, new_adj = gae.train(ut.graph, model_name, ut.input, ut.input_size, layer_size, layer_size,
                                       250,
                                       10, False)
    # save embedding
    np.save(ut.embedding_path + model_name + "_embedding.npy", embedding)
    return embedding


def matrix_factorization(model_name):
    """generate embedding by laplacian eigenmaps"""

    embedding = spectral_embedding(nx.adjacency_matrix(ut.graph), n_components=64)

    # save embedding
    np.save(ut.embedding_path + model_name + "_embedding.npy", embedding)
    return embedding


def node2vec(model_name):
    """generate embedding with node2vec"""
    if model_name == "Nove2Vec_Structural":
        p = 0.5
        q = 2
    elif model_name == "Nod2Vec_Homophily":
        p = 1
        q = 0.5
    node2vec = Node2Vec(ut.graph, dimensions=64, walk_length=80, num_walks=10, workers=4, p=p, q=q)
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,
                         batch_words=4)

    # save embedding
    np.save(ut.embedding_path + model_name + "_embedding.npy", model.wv.vectors)
    return model.wv.vectors
