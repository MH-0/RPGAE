#  Exploring the Representational Power of Graph Autoencoders


In this study, we look into the representational power of Graph Autoencoders (GAE) and verify if the following topological features are being captured in the embeddings: Degree, Local Clustering Score, Eigenvector Centrality, Betweenness Centrality. We also verify if the presence of these topological features leads to better performance on the downstream tasks of node clustering and classification. Our experimental results show that these topological features, especially the degree, are indeed preserved in the **first layer** of the GAE that employs the **SUM** aggregation rule, under the condition that the model **preserves the second order proximity**. We also show that a model with such properties can outperform other models on certain downstream tasks, especially when the preserved features are relevant to the task at hand.  Finally, we evaluate the suitability of our findingsthrough a test case study related to social influence prediction.

## Getting Started

For the complete list of results please check the **"results/"** folder.

### Prerequisites
- python 3.6
- dgl-cu90 0.4.3
- torch 1.4
- networkx 2.4
- numpy 1.18.1
- scikit-learn 0.22.2
- node2vec 0.3.2
- cuda 9.0

### Experimental Datasets

|Dataset                 |Nodes |Edges |Classes |Type          |BINS | Reference |
| -                      |-     |     -|-       |-             |-    |- |
|Cora                    |2,708 |5,429 |7       |Citation      |6|[2] |
|Citeseer                |3,327 |4,732 |6       |Citation      |6| [2]|
|email-Eu-core           |1,005 |25,571|42      |Email         |6| [1]|
|USA Air-Traffic         |1,190 |13,599|4       |Flight        |3| [3]| 
|Europe Air-Traffic      |399   |5,995 |4       |Flight        |3| [3]| 
|Brazil Air-Traffic      |131   |1,074  |4       |Flight        |3| [3]|
|fly-drosophila-medulla-1|1,781 |9,016 |NA      |Biological    |6| [1]|
|ego-Facebook            |4,039 |88,234|NA      |Social        |6| [1]|
|soc-sign-bitcoin-alpha  |3,783 |24,186|NA      |Blockchain    |6| [1]|
|soc-sign-bitcoin-otc    |5,881 |35,592|NA      |Blockchain    |6| [1]|
|ca-GrQc                 |5,242 |14,496|NA      |Collaboration |4| [1]|

### Compared Models
|Model | Description|
|-     | -           |
|GAE_L1_SUM | A Graph Autoencoder with 1 layer that employs the SUM aggregation rule|
|GAE_L2_SUM | A Graph Autoencoder with 2 layers that employs the SUM aggregation rule|
|GAE_FIRST | A Graph Autoencoder that employs the SUM aggregation rule but uses the output of the first layer as embedding|
|GAE_CONCAT| A Graph Autoencoder that employs the SUM aggregation rule but concatenates the output of all layers as embedding|
|GAE_MEAN|A Graph Autoencoder that employs the MEAN aggregation rule|
|GAE_SPECTRAL|A Graph Autoencoder that employs the SPECTRAL rule (GCN)|
|GAE_MIXED| A Graph Autoencoder that employes the MEAN aggregation rule but that reconstructs 2 orders of proximity|
|Matrix Factorization| Laplacian eigenmaps (scikit-learn implementation)|
|Nove2Vec_Structural| Node2vec with p=0.5 q =2|
|Nod2Vec_Homophily| Node2vec with p=1 q=0.5| 

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

### Running a Custom Experiment
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

### Loading a Custom Dataset
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

### References
[1] J.  Leskovec  and  A.  Krevl,  “SNAP  Datasets:  Stanford  large  networkdataset collection.” http://snap.stanford.edu/data, June 2014.

[2] M.  Wang,  L.  Yu,  D.  Zheng,  Q.  Gan,  Y.  Gai,  Z.  Ye,  M.  Li,  J.  Zhou,Q. Huang, C. Ma, Z. Huang, Q. Guo, H. Zhang, H. Lin, J. Zhao, J. Li,A. J. Smola, and Z. Zhang, “Deep graph library: Towards efficient andscalable deep learning on graphs,” inICLR Workshop on RepresentationLearning on Graphs and Manifolds, 2019.

[3] J.  Wu,  J.  He,  and  J.  Xu,  “Demo-net:  Degree-specific  graph  neuralnetworks for node and graph classification,” inProceedings of the 25thACM  SIGKDD  International  Conference  on  Knowledge  Discovery  &Data Mining, pp. 406–415, 2019.
