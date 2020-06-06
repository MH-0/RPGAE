#  Exploring the Representational Power of Graph Autoencoders

One Paragraph of project description goes here

## Getting Started

For the complete list of results please check the **"results/"** folder

### Prerequisites
- python 3.6
- dgl-cu90 0.4.3
- torch 1.4
- networkx 2.4
- numpy 1.18.1
- scikit-learn 0.22.2
- node2vec 0.3.2
- cuda 9.0

## Running Experiments

In order to run a test on a dataset, exectute the below code snippet:
```
import experiments.run_experiments as re
re.run(dataset_name, bins, iterations)
```

- **dataset_name**: Name of the dataset to be tested, as defined in the  **"data\"** folder
- **bins**: The number of bins (classes) to split the continuous topological features into
- **iterations**: The number of times to iterate the experiments (the results will be the mean)

Example:
```
import experiments.run_experiments as re
re.run("brazil_airtraffic", 3, 1)
```

### Running Custom Experiment
```
import evaluate_embeddings.visualize_embeddings as ve
import experiments.utils as eu
import pretreatment.topo_features as tf
import pretreatment.utils as ut

ut.load_dataset(dataset_name)
  
tf.save_topo_features()

tf.generate_features_labels(bins)

eu.iterate_experiments(
    ["gae_l1_sum", "gae_l2_sum", "gae_concat", "gae_first", "gae_mean", "gae_mixed", "gae_spectral",
     "matrix_factorization", "Nove2Vec_Structural", "Nod2Vec_Homophily"], iterations)
     
ve.visualize_results()
```
- **load_dataset**: Loads a predefined dataset, the dataset name should be as defined in the  **"data\"** folder
- **save_topo_features**: Calculates the topological features of the vertices
- **generate_features_labels**: Performs the binning operation, according to the number of **bins** defined by the user
- **iterate_experiments**: Takes as a parameter the list of embedding models to be tested and the number of iterations to run the experiments
- **visualize_results**: Plots the embeddings of 4 models: gae_first, gae_concat, gae_l1_sum ,matrix_factorization

### Loading Custom Dataset
Place the new dataset folder in the **"data\"** folder, the graph files should be placed in a sub folder named **"graph"**

example : data\newdataset\graph\

3 files can be placed in the **"graph\"** folder. All files can be comma "," or tab"\t" or  space " " seperated 

The same seperator should be used in all 3 files
- **edges.txt**: A text file containing the list of edges  "Node1 sperator Node2" (Mandatory)
- **groundtruth.txt**: A text file containing the list of classes "Node Class" (if the vertices have a ground truth)
- **attributes.txt**: A text file containing the list of attributes " Node Attribute 1, Attribute2, ..." (if the vertices are attributed)

In order to load a custom dataset use the function **load_custom_dataset** instead of **load_dataset**

```
load_custom_dataset(dataset_name, with_attributes, with_labels, directed, separator)
```
- **dataset_name**: the name of the custom dataset
- **with_attributes**: if the graph is attributed
- **with_labels**: if the vertices have ground truth
- **directed**: if the graph is directed 
- **separator**: the seperator " " or "\t" or ","

Example:

```
import pretreatment.utils as ut
ut.load_custom_dataset("europe_airtraffic", False, True, False, " ")
```
