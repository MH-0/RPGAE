""" Graph auto-encoder that reconstructs both
    1st and 3rd degree order adjacency matrix
    it uses the mean aggregation rule
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import networkx as nx

# message passing function
gcn_message = fn.copy_src(src='h', out='m')

# reduce (aggregation) function (uses mean of the attributes)
gcn_mean_reduce = fn.mean(msg='m', out='h')

# custom message passing function (not used)
def gcn_message_custom(edges):
    return {'m': edges.src['h'], 'w': edges.data['weight'].float(), 's': edges.src['deg'], 'd': edges.dst['deg']}

# custom aggregation  function (not used)
def gcn_reduce_custom(nodes):
    return {
        'h': torch.sum(nodes.mailbox['m'] * nodes.mailbox['w'].unsqueeze(2), dim=1)}

class EncoderLayer(nn.Module):
    """
    Encoder layer of the auto-encoder
    """
    def __init__(self, in_feats, out_feats, activation, dropout):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation
        self.norm = nn.BatchNorm1d(out_feats)
        self.drop = nn.Dropout(dropout)

    def forward(self, g: dgl.graph, input):
        g.ndata['h'] = input
        g.update_all(gcn_message, gcn_mean_reduce)

        h = g.ndata.pop('h')
        h = self.linear(h)
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

    def forward(self, input):
        h = F.softmax(input, 1)
        return h


class DecoderLayer(nn.Module):
    def __init__(self, activation, num_features, dropout):
        super(DecoderLayer, self).__init__()
        self.activation = activation
        self.var = torch.var
        self.norm = nn.BatchNorm1d(num_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        # the decoder reconstructs the adjacency by multiplying
        # the output of the encoder with its transpose
        h = torch.mm(z, z.t())
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h


class GAE(nn.Module):
    def __init__(self, number_nodes, input_size, hidden_size, encoded_size):
        super(GAE, self).__init__()
        self.enc1 = EncoderLayer(input_size, hidden_size, torch.tanh, 0)
        self.enc2 = EncoderLayer(hidden_size, encoded_size, torch.tanh, 0)

        self.dec = DecoderLayer(torch.sigmoid, number_nodes, 0)
        self.clas = classifier()

    def forward(self, g, inputs):
        # Encoder
        encoded1 = self.enc1.forward(g, inputs)
        encoded2 = self.enc2.forward(g, encoded1)

        # Decoder
        # Decode first level of proximity
        decoded = self.dec.forward(encoded1)
        # Decode second level of proximity
        decoded2 = self.dec.forward(encoded2)

        return encoded2, decoded, decoded2


def train(graph, inputs, input_size, hidden_size, embedding_size, epochs, early_stopping,
          print_progress=True):
    """
    This function trains the graph autoencoder in order to generate the embeddings of the graph (a vector per node)
    :param graph: a networkx graph for a time step
    :param model_name: the name of the model (gae_sum,gae_mean,gae_spectral,gae_sum_concat)
    :param inputs: the attributes to be used as input
    :param input_size: the size of the input
    :param hidden_size: the hidden layer size
    :param embedding_size: the embedding (encoder output) size
    :param epochs: the number of training epochs
    :param early_stopping: the number of epochs for early stopping
    :param print_progress: whether to print the training progress or not
    :return: the embedding of the graph (a vector for every node in the graph)
    """

    # generate a dgl graph object from the networkx object
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(graph)

    # adding self loops
    dgl_graph.add_edges(dgl_graph.nodes(), dgl_graph.nodes())

    # adjacency matrix (for reconstruction)
    adjacency_ground_truth = torch.tensor(dgl_graph.adjacency_matrix().to_dense())
    # adjacency matrix second order (for reconstruction)

    # g = kneighbors_graph(inputs, 100, mode='distance', include_self=True)
    # knndgl = dgl.DGLGraph()
    # graphh= nx.from_scipy_sparse_matrix(g)
    # knndgl.from_networkx(graphh)
    # adjacency_ground_truth2 = torch.tensor(knndgl.adjacency_matrix().to_dense())
    adjacency_ground_truth2 = torch.pow(adjacency_ground_truth.clone(), 3)

    # node feature (the degree of the node)
    dgl_graph.ndata['deg'] = dgl_graph.out_degrees(dgl_graph.nodes()).float()

    # model
    gae = GAE(graph.number_of_nodes(), input_size, hidden_size, embedding_size)

    # for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gae = gae.to(device)
    dgl_graph.to(device)
    inputs = inputs.to(device)
    adjacency_ground_truth = adjacency_ground_truth.to(device)
    adjacency_ground_truth2 = adjacency_ground_truth2.to(device)
    # optimizer used for training
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)

    min_loss = 1000
    stop_index = 0
    alpha = 0.5
    # iterate through epochs
    for epoch in range(epochs):
        embedding, predicted, predicted2 = gae.forward(dgl_graph, inputs)
        embedding = embedding.to(device)
        predicted = predicted.to(device)
        predicted2 = predicted2.to(device)

        # Loss of 1st order
        loss_reconstruct = F.mse_loss(predicted, adjacency_ground_truth)
        # loss of 3rd order
        loss_reconstruct2 = F.mse_loss(predicted2, adjacency_ground_truth2)

        loss = loss_reconstruct + alpha * loss_reconstruct2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < min_loss:
            if print_progress:
                print('Epoch %d | old Loss: %.4f | New Loss:  %.4f ' % (epoch, min_loss, loss.item()))

            # we only save the embedding if there is an improvement in training
            save_emb = embedding
            save_pred = predicted
            min_loss = loss
            stop_index = 0
        else:
            if print_progress:
                print('Epoch %d | No improvement | Loss: %.4f | old Loss :  %.4f ' % (epoch, loss.item(), min_loss))
            stop_index += 1

        if stop_index == early_stopping:
            if print_progress:
                print("Early Stopping!")
            break

    save_emb = save_emb.detach().cpu()
    save_emb = save_emb.numpy()
    save_pred = torch.round(save_pred).detach().cpu()
    save_pred = save_pred.numpy()
    adjacency_ground_truth = adjacency_ground_truth.cpu()

    return save_emb

