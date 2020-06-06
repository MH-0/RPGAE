import pretreatment.utils as ut
import pretreatment.topo_features as tf
import experiments.utils as eu
import evaluate_embeddings.visualize_embeddings as ve

# load the graph data into memory
ut. load_dataset("cora")
# # generate topological features
# # ------------------------------
# tf.save_topo_features()

# # binning topological features
# # -----------------------------
# tf.generate_features_labels(6)

# run experiments
# ------------------
# eu.iterate_experiments([ "gae_first"], 1)

# eu.iterate_experiments(["gae_one", "gae_sum", "gae_sum_concat", "gae_first", "gae_mean", "gae_mixed", "gae_spectral",
#                         "matrix_factorization","Nove2Vec_Structural","Nod2Vec_Homophily"], 10)

# # visualize results
# # ----------------------

# ve.visualize_results()
ve.visualize_Large("gae_first","degree")