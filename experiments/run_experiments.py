import experiments.scores as sc
import experiments.utils as eu
import pretreatment.topo_features as tf
import pretreatment.utils as ut
import evaluate_embeddings.visualize_embeddings as ve

def run(dataset_name, bins, iterations):
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
