import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e': F.leaky_relu(a)}


def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h': h}

class DecoderLayer(nn.Module):
    def __init__(self, activation, num_features):
        super(DecoderLayer, self).__init__()
        self.activation = activation
        self.var = torch.var
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        # the decoder reconstructs the adjacency by mulitplying the output of the
        # encoder with its transpose
        h = torch.mm(inputs, inputs.t())
        h = self.activation(h)
        h = self.norm(h)
        return h

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, number_nodes, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim,1)
        self.dec = DecoderLayer(torch.sigmoid, number_nodes)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        decoded = self.dec.forward(h)

        return decoded, h


def train(graph, inputs, input_size, hidden_size, embedding_size, epochs, early_stopping,
          print_progress=True):
    """
    This function trains the graph autoencoder in order to generate the embeddings of the graph (a vector per node)
    :param graph: a networkx graph for a time step
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

    # adjacency matrix
    adj = torch.tensor(dgl_graph.adjacency_matrix().to_dense())

    gae = GAT(dgl_graph,graph.number_of_nodes(), input_size, hidden_size, embedding_size,2)

    # for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gae = gae.to(device)
    dgl_graph.to(device)
    inputs = inputs.to(device)
    adj = adj.to(device)

    # optimizer used for training
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.001)

    min_loss = 1000
    stop_index = 0
    for epoch in range(epochs):
        predicted, embedding = gae.forward(inputs)
        embedding = embedding.to(device)
        predicted = predicted.to(device)
        loss = F.mse_loss(predicted, adj)

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
    adj = adj.cpu()

    return save_emb