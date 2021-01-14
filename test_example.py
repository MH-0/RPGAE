import experiments.run_experiments as re

import experiments.scores as sc
import experiments.utils as eu
import pretreatment.utils as ut
import evaluate_embeddings.visualize_embeddings as ve
import pretreatment.topo_features as tf

runs = 1

# re.run("cora", 6, runs)
# re.run("citeseer", 6, runs)
# re.run("eumails", 6, runs)
#
# re.run("usa_airtraffic", 3, runs)
# re.run("europe_airtraffic", 3, runs)
# re.run("brazil_airtraffic", 3, runs)
#
# re.run("flydrosophilamedulla", 6, runs)
# re.run("facebook", 6, runs)
# re.run("socsignbitcoinalpha", 6, runs)
# re.run("socsignbitcoinot", 6, runs)
# re.run("ca-grqc", 4, runs)

dataset_name = "cora"
ut.load_dataset(dataset_name)
# # eu.iterate_experiments(
# #         ["gae_first", "gae_concat", "gae_mixed", "gae_l1_sum", "matrix_factorization"], runs)
#
# ve.visualize_results(True)
tf.generate_features_labels(6)



