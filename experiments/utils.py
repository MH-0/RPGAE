"""
this file contains a list of functions used to run the experiments
"""
import time

import embedding_models.generate_embeddings as ge
import evaluate_embeddings.classify_embeddings as ce
import evaluate_embeddings.classify_features as cf
import evaluate_embeddings.cluster_embeddings as cl
import evaluate_embeddings.cluster_similarity as cs
import pretreatment.utils as ut

def iterate_experiments(list_embedding_model_name, iterations):
    """
    iterate through the experiments for a list of models
    :param list_embedding_model_name: list of model names
    :param iterations: number of iterations to run the experiments
    """

    # track run time
    start = time.time()

    # loop through models
    for embedding_model_name in list_embedding_model_name:
        experiment_by_model(embedding_model_name, iterations)

    # end clock
    end = time.time()

    # print scores
    cf.print_score()
    ce.print_score()
    cl.print_score()
    cs.print_score()

    print()
    print()
    print("Done! Experiments", ut.get_elapsed(start, end))


def experiment_by_model(embedding_model_name, iterations):
    """
    run experiments by 1 model
    :param embedding_model_name: model name
    :param iterations: number of iterations to run the experiments
    :return:
    """

    # setup dictionary scores
    # for topo features
    cf.setup_score(embedding_model_name, "degree")
    cf.setup_score(embedding_model_name, "clustering")
    cf.setup_score(embedding_model_name, "eigenvector_centrality")
    cf.setup_score(embedding_model_name, "betweenness_centrality")

    # for classification of labels
    ce.setup_score(embedding_model_name)
    # for clustering
    cl.setup_score(embedding_model_name)
    # for cluster evaluation
    cs.setup_score(embedding_model_name)

    # iterate through number of experiments
    for iteration in range(0, iterations):
        print("experiment:", iteration + 1, " Model:", embedding_model_name)

        print("generating embeddings...")

        if embedding_model_name == "matrix_factorization":
            ge.matrix_factorization(embedding_model_name)
        elif embedding_model_name == "Nove2Vec_Structural":
            ge.node2vec(embedding_model_name)
        elif embedding_model_name == "Nod2Vec_Homophily":
            ge.node2vec(embedding_model_name)
        else:
            ge.gae_embedding(embedding_model_name)

        print("classifying features...")
        cf.classify_features(embedding_model_name, "degree")
        cf.classify_features(embedding_model_name, "clustering")
        cf.classify_features(embedding_model_name, "eigenvector_centrality")
        cf.classify_features(embedding_model_name, "betweenness_centrality")

        # if graph has ground truth
        if ut.number_classes != 0:
            print("classifying nodes...")
            ce.classify_embeddings(embedding_model_name)

            print("cluster nodes...")
            cl.cluster_embeddings(embedding_model_name)

            print("cluster similarity...")
            cs.calculate_similarity(embedding_model_name)

