# Added for the paper "Exploring the representational power of Graph Autoencoder"
"""
This file is used to the
"""

import torch
torch.cuda.current_device()
import train as tr

dataset = "digg"

# this variable should be modified to the path of the deepInf datasets
data_directory = "...\\deepinfodata\\"

data_path =  data_directory + dataset

print(dataset)

# These functions train each of the compared model variations
# Each function should be executed separately
# ------------------------------------------------

tr.train_evaluate_model(data_path,"gnn_sum","concat","128,128",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_sum","concat","64,64",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_sum","l1","128",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_sum","l2","128,128",False,False,False)
# tr.train_evaluate_model(data_path,"gcn","","128,128",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_mean","","128,128",False,False,False)
