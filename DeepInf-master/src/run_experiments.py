import torch
torch.cuda.current_device()
import train as tr

dataset = "digg"

data_directory = "D:\\UNIVERSITY\\Masters\\Memoire\\Exploring the representational power of graph autoencoders\\github\\deepinfodata\\"
data_path =  data_directory + dataset

print(dataset)
# tr.train_evaluate_model(data_path,"gnn_sum","concat","128,128",False,False,False)
tr.train_evaluate_model(data_path,"gnn_sum","concat","64,64",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_sum","l1","128",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_sum","l2","128,128",False,False,False)
# tr.train_evaluate_model(data_path,"gcn","","128,128",False,False,False)
# tr.train_evaluate_model(data_path,"gnn_mean","","128,128",False,False,False)
