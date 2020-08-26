"""
Graph autoencoder models
6 variations:
-gae_l1_sum (aggregation by sum of the features - with 1 layer encoder)
-gae_l2_sum (aggregation by sum of the features - with 2 layers encoder)
-gae_first (aggregation by sum but with first layer output as embedding)
-gae_concat (aggregation by sum but with concatenation of the output of all layers as embedding)
-gae_mean (aggregation by mean of the features)
-gae_spectral (applies the spectral filter)
"""

# load libraries
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# message passing function
gcn_message = fn.copy_src(src='h', out='m')

# reduce (aggregation) function (uses sum of the attributes)
gcn_sum_reduce = fn.sum(msg='m', out='h')

# reduce (aggregation) function (uses mean of the attributes)
gcn_mean_reduce = fn.mean(msg='m', out='h')


# custom message passing function (not used just for testing)
def gcn_message_custom(edges):
    return {'m': edges.src['h'], 'w': edges.data['weight'].float(), 's': edges.src['deg'], 'd': edges.dst['deg']}


# custom aggregation  function (not used just for testing)
def gcn_reduce_custom(nodes):
    return {
        'h': torch.sum(nodes.mailbox['m'] * nodes.mailbox['w'].unsqueeze(2) / nodes.mailbox['d'].unsqueeze(2), dim=1)}


network_type = ""

class EncoderLayer(nn.Module):
    """
    The encoder layers of the auto-encoder
    """

    def __init__(self, in_feats, out_feats, activation, dropout):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation
        self.norm = nn.BatchNorm1d(out_feats)
        self.drop = nn.Dropout(dropout)

    def forward(self, g: dgl.graph, input):
        g.ndata['h'] = input
        if network_type == "gae_mean":
            # broadcast all messages and aggregate them according to the mean rule
            g.update_all(gcn_message, gcn_mean_reduce)
        else:
            # broadcast all messages and aggregate them according to the sum rule
            g.update_all(gcn_message, gcn_sum_reduce)

        h = g.ndata.pop('h')
        h = self.linear(h)
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h

class DecoderLayer(nn.Module):
    """
    The decoder layer of the auto-encoder
    """

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
        # for the spectral rule use the layers given by DGL library
        self.enc_spectral_1 = GraphConv(input_size, hidden_size, norm="both", bias=True, activation=F.tanh)
        self.enc_spectral_2 = GraphConv(hidden_size, encoded_size, norm="both", bias=True, activation=F.tanh)
        # for the other rules use the custom layers
        self.enc_custom_1 = EncoderLayer(input_size, hidden_size, torch.tanh, 0)
        self.enc_custom_2 = EncoderLayer(hidden_size, encoded_size, torch.tanh, 0)

        self.dec = DecoderLayer(torch.sigmoid, number_nodes, 0)

    def forward_spectral(self, g, inputs):
        # encoder
        encoded1 = self.enc_spectral_1.forward(g, inputs)
        encoded2 = self.enc_spectral_2.forward(g, encoded1)
        # decoder
        decoded = self.dec.forward(encoded2)

        embedding = encoded2
        return decoded, embedding

    def forward_1_layer(self, g, inputs):
        # encoder
        encoded1 = self.enc_custom_1.forward(g, inputs)
        # decoder
        decoded = self.dec.forward(encoded1)

        embedding = encoded1
        return decoded, embedding

    def forward_2_layers(self, g, inputs):
        # encoder
        encoded1 = self.enc_custom_1.forward(g, inputs)
        encoded2 = self.enc_custom_2.forward(g, encoded1)
        # decoder
        decoded = self.dec.forward(encoded2)

        embedding = encoded2
        return decoded, embedding

    def forward_first_layer(self, g, inputs):
        # encoder
        encoded1 = self.enc_custom_1.forward(g, inputs)
        encoded2 = self.enc_custom_2.forward(g, encoded1)
        # decoder
        decoded = self.dec.forward(encoded2)

        embedding = encoded1
        return decoded, embedding

    def forward_concat_layer(self, g, inputs):
        # encoder
        encoded1 = self.enc_custom_1.forward(g, inputs)
        encoded2 = self.enc_custom_2.forward(g, encoded1)
        # decoder
        decoded = self.dec.forward(encoded2)

        embedding = torch.cat((encoded1, encoded2),
                              dim=1)
        return decoded, embedding


def train(graph, model_name, inputs, input_size, hidden_size, embedding_size, epochs, early_stopping,
          print_progress=True):
    """
    This function trains the graph autoencoder in order to generate the embeddings of the graph (a vector per node)
    :param graph: a networkx graph for a time step
    :param model_name: the name of the model (gae_l1_sum,gae_l2_sum,gae_mean,gae_spectral,gae_concat,gae_first)
    :param inputs: the attributes to be used as input
    :param input_size: the size of the input
    :param hidden_size: the hidden layer size
    :param embedding_size: the embedding (encoder output) size
    :param epochs: the number of training epochs
    :param early_stopping: the number of epochs for early stopping
    :param print_progress: whether to print the training progress or not
    :return: the embedding of the graph (a vector for every node in the graph)
    """
    global network_type
    # input_size = 1
    # inputs = torch.rand(len(graph.nodes), input_size)
    # print(inputs)
    # inputs =  torch.eye(len(graph.nodes))

    network_type = model_name

    # generate a dgl graph object from the networkx object
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(graph)

    # adding self loops
    dgl_graph.add_edges(dgl_graph.nodes(), dgl_graph.nodes())

    # adjacency matrix (for reconstruction)
    adjacency_ground_truth = torch.tensor(dgl_graph.adjacency_matrix().to_dense())

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

    # optimizer used for training
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)

    min_loss = 1000
    stop_index = 0
    # iterate through epochs
    for epoch in range(epochs):
        if network_type == "gae_spectral":
            predicted, embedding = gae.forward_spectral(dgl_graph, inputs)
        elif network_type == "gae_l1_sum":
            predicted, embedding = gae.forward_1_layer(dgl_graph, inputs)
        elif network_type == "gae_first":
            predicted, embedding = gae.forward_first_layer(dgl_graph, inputs)
        elif network_type == "gae_concat":
            predicted, embedding = gae.forward_concat_layer(dgl_graph, inputs)
        else:
            predicted, embedding = gae.forward_2_layers(dgl_graph, inputs)

        embedding = embedding.to(device)
        predicted = predicted.to(device)

        loss = F.mse_loss(predicted, adjacency_ground_truth)

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

    return save_emb, save_pred
