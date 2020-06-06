#  Exploring the Representational Power of Graph Autoencoders

One Paragraph of project description goes here

## Getting Started

For the complete list of results please check the **results/** folder

### Prerequisites
- python 3.6
- dgl-cu90 0.4.3
- torch 1.4
- networkx 2.4
- numpy 1.18.1
- scikit-learn 0.22.2
- node2vec 0.3.2
- Cuda 9.0

## Running the tests

In order to run a test on a dataset, exectute the below code snippet:
```
import experiments.run_experiments as re
re.run(dataset_name, bins, iterations)
```

- dataset_name: name of the dataset to be tested as defined in the  **data\** folder
- bins: the number of bins (classes) to split the continuous topological features into
- iterations: the number of times to iterate the experiments (the results will be the mean)

Example:
import experiments.run_experiments as re
re.run("brazil_airtraffic", 3, 1)


### In order to run a custom test

### In order to load a custom dataset


