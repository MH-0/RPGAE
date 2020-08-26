"""
This file contains miscellaneous functions used in loading and pre-treating data
"""
from pathlib import Path

import dgl.data.citation_graph as cg
import dgl.data.gnn_benckmark as gn
import networkx as nx
import numpy as np
import torch

data_path = ""
graph_path = ""
topo_features_path = ""
topo_features_labels_path = ""
embedding_path = ""
graph = nx.Graph()
node_labels = []
number_classes = 0
input = np.array(0)
input_size = 0

def load_graph(edges_file_path, top_lines_to_remove, split_char='\t', nodes_file_path='', print_details=False,
               directed=True):
    """
    this function loads a graph file into memory as a networkx graph object
    the graph file should be a list of edges of the following format:

    node1 "Separator" node2

    :param edges_file_path: the path to the file containing the list edges
    :param top_lines_to_remove: the number of lines at the top of the file to be ignored
    :param split_char: the separation character between two nodes (default is TAB)
    :param nodes_file_path: the path to the file containing the list nodes (optional - empty if not present)
    :param print_details: print the details of the graph (number of nodes and number of edges)
    :return: Networkx graph object
    """

    edges_file = open(edges_file_path, "r")
    data = edges_file.read().split("\n")

    # remove top lines
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)

    if nodes_file_path:
        nodes_file = open(nodes_file_path, "r")
        nodes_data = nodes_file.read().split("\n")
        nodes = [int(i) for i in nodes_data if i != ""]
        nodes = sorted(set(nodes))
    else:
        # if the node file is not present, extract the nodes from the edges
        nodes1 = [int(i.split(split_char, 1)[0]) for i in data if i != ""]
        nodes2 = [int(i.split(split_char, 1)[1]) for i in data if i != ""]

        nodes = sorted(set(nodes1 + nodes2))

    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    g.add_nodes_from(nodes)

    # adding edges to the graph
    for index in range(0, len(data)):
        adjacent_nodes = data[index].split(split_char)
        if adjacent_nodes[0] == "":
            continue
        if adjacent_nodes[1] == "":
            continue
        node_1 = int(adjacent_nodes[0])
        node_2 = int(adjacent_nodes[1])

        g.add_edge(node_1, node_2)

    if print_details:
        print("graph loaded:")
        print("-------------")
        print("nodes:" + str(g.number_of_nodes()))
        print("edges:" + str(g.number_of_edges()))
        print("-------------")

    return g


def load_attributes(file_path, top_lines_to_remove, split_char=','):
    """
    load an attribute file related to a graph into memory
    :param file_path: the path of the attribute file
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :param split_char: separation character
    :return:list of attributes of the graph
    """
    file = open(file_path, "r")
    data = file.read().split("\n")
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)
    all_attributes = []
    for line in data:
        if line == "": continue
        attributes = line.split(split_char)

        for index in range(0, len(attributes)):
            attributes[index] = attributes[index].replace('[', '').replace(']', '').replace(' ', '')
            attributes[index] = float(attributes[index])

        all_attributes.append(attributes)
    return all_attributes


def load_groundtruth(file_path, top_lines_to_remove, split_char='\t'):
    """
    load the classes of the nodes
    :param file_path: he path of the classes file
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :param split_char: separation character
    :return:
    """
    ground_truth_file = open(file_path, "r")
    data = ground_truth_file.read().split("\n")
    ground_truth = []

    # remove top lines
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)

    for index in range(0, len(data)):
        nodes_class = data[index].split(split_char)
        if nodes_class[0] == "":
            continue
        if nodes_class[1] == "":
            continue
        ground_truth.append(int(nodes_class[1]))
    return ground_truth


def get_elapsed(start, end):
    """
    used to calculate the time between two time stamps

    :param start: start time
    :param end: end time
    :return: a string in minutes or seconds for the elapsed time
    """
    elapsed = end - start
    if elapsed < 60:
        return '{0:.2g}'.format(end - start) + " seconds"
    else:
        return '{0:.2g}'.format((end - start) / 60.0) + " minutes"


def print_list(lst, lst_name):
    """
    standing list printing
    :param lst: the list to be printed
    :param lst_name: the title of the list
    :return:
    """
    print("----------")
    print(lst_name)
    print("----------")
    index = 0
    for series in lst:
        print(index, " \t size: " + str(len(series)), series)
        index += 1


def load_numpy_file(file_path):
    """
    load a numpy file into memory
    :param file_path: numpy file path
    :return: numpy array
    """
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    numpy_array = np.load(file_path)
    np.load = np_load_old
    return numpy_array


def load_custom_dataset(dataset_name, with_attributes, with_labels, directed, separator):
    """
    loads the dataset into memory
    :param dataset_name: The name of the dataset
    :param with_attributes: if it has attributes
    :param separator: the separator character in the files (" " or "," or "\t")
    """
    global data_path
    global graph_path
    global topo_features_path
    global topo_features_labels_path
    global embedding_path
    global graph
    global node_labels
    global number_classes
    global input
    global input_size

    # data folder path
    data_path = "data\\" + dataset_name + "\\"

    # graph folder path
    graph_path = data_path + "graph\\"

    # features folder path
    topo_features_path = data_path + "top_features\\"

    # features classes folder path
    topo_features_labels_path = data_path + "top_features_labels\\"

    # pretreatment folder path
    embedding_path = data_path + "embedding\\"

    # Load graphs
    if dataset_name == "cora":
        data = cg.load_cora()
        graph = data.graph
        node_labels = data.labels
        input = torch.tensor(data.features).float()
        # input = torch.FloatTensor([1]*len(graph.nodes)).reshape(len(node_labels),1)
        # input = torch.eye(len(graph.nodes))
    elif dataset_name == "citeseer":
        data = cg.load_citeseer()
        graph = data.graph
        node_labels = data.labels
        input = torch.tensor(data.features).float()
    else:
        graph = load_graph(graph_path + "edges.txt", 0, separator, print_details=True, directed=directed)
        if with_labels:
            node_labels = load_groundtruth(graph_path + "groundtruth.txt", 0, separator)
        else:
            node_labels = []
        if with_attributes:
            input = load_attributes(graph_path + "attributes.txt", 0, separator)
        else:
            input = torch.eye(len(graph.nodes))

    # input layer size
    input_size = len(input[0])

    # number of classes for the node labels
    number_classes = len(set(node_labels))

    # create directories if they do not exist
    # folder that holds the embeddings
    Path(embedding_path).mkdir(parents=True, exist_ok=True)
    # folder that holds the topological features
    Path(topo_features_path).mkdir(parents=True, exist_ok=True)
    # folder that holds the classes of the topological features
    Path(topo_features_labels_path).mkdir(parents=True, exist_ok=True)
    print("graph details:", dataset_name)
    print("------------------")
    print("nodes", len(graph.nodes))
    print("edges", len(graph.edges))
    print("classes", len(set(node_labels)))
    print("------------------")


def load_dataset(dataset_name):
    if dataset_name == "cora":
        load_custom_dataset("cora", True, True, False, "")
    elif dataset_name == "citeseer":
        load_custom_dataset("citeseer", True, True, False, "")
    elif dataset_name == "eumails":
        load_custom_dataset("eumails", False, True, True, " ")
    elif dataset_name == "facebook":
        load_custom_dataset("facebook", False, False, False, " ")
    elif dataset_name == "terroristrel":
        load_custom_dataset("terroristrel", False, True, True, ",")
    elif dataset_name == "flydrosophilamedulla":
        load_custom_dataset("flydrosophilamedulla", False, False, False, " ")
    elif dataset_name == "socsignbitcoinalpha":
        load_custom_dataset("socsignbitcoinalpha", False, False, False, ",")
    elif dataset_name == "socsignbitcoinot":
        load_custom_dataset("socsignbitcoinot", False, False, False, ",")
    elif dataset_name == "ca-grqc":
        load_custom_dataset("ca-grqc", False, False, False, "\t")
    elif dataset_name == "usa_airtraffic":
        load_custom_dataset("usa_airtraffic", False, True, False, ",")
    elif dataset_name == "brazil_airtraffic":
        load_custom_dataset("brazil_airtraffic", False, True, False, " ")
    elif dataset_name == "europe_airtraffic":
        load_custom_dataset("europe_airtraffic", False, True, False, " ")
