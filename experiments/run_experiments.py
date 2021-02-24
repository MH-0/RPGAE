"""
This file is used to run the set of all experiments per dataset
"""

# load libraries
import experiments.scores as sc
import experiments.utils as eu
import pretreatment.topo_features as tf
import pretreatment.utils as ut
import evaluate_embeddings.visualize_embeddings as ve

def run(dataset_name, bins, iterations):
    """
    This function runs the set of experiments on a particular dataset.
    1- It loads the dataset
    2- it calculates the topological features
    3- it generates the topologcail feature classes (binning)
    4- it iterates the experiments for every embedding model
    5- it visualises the results (2D plots)
    6- it displays the scores of the experiments
    :param dataset_name: the name of the dataset on which to run the experiments
    :param bins: the number of bins to split the topological features into
    :param iterations: the number of iteration runs
    """

    # load the graph data into memory
    # ------------------------------
    ut.load_dataset(dataset_name)

    # generate topological features
    # ------------------------------
    tf.save_topo_features()

    # binning topological features
    # -----------------------------
    tf.generate_features_labels(bins)

    # run experiments
    # ------------------
    eu.iterate_experiments(
        ["gae_first", "gae_concat", "gae_l1_sum", "gae_l2_sum", "gae_mean", "gae_mixed", "gae_spectral",
         "matrix_factorization", "Nove2Vec_Structural", "Nod2Vec_Homophily"], iterations)

    # visualize results
    # ----------------------
    ve.visualize_results()

    # print results
    # ----------------------
    sc.print_scores(dataset_name)
