import torch
import torch.nn as nn
import torch.nn.functional as F




class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        # x: Node features (batch_size, num_nodes, in_features)
        # adjacency_matrix: Graph adjacency matrix (batch_size, num_nodes, num_nodes)

        # Normalize adjacency matrix
        # adjacency_matrix = F.normalize(adjacency_matrix, p=1, dim=2)

        # Perform graph convolution
        x = torch.bmm(adjacency_matrix, x)
        x = self.linear(x)

        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.layer2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = self.layer1(x, adjacency_matrix)
        x = self.layer2(x, adjacency_matrix)
        return x



import torch
import math
from torch import nn

class GraphConvolution2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_dim, out_features_dim, activation=None, bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.news.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # self.bias.news.uniform_(-stdv, stdv)
            nn.init.zeros_(self.bias)

    def forward(self, infeatn, adj):
        '''
        infeatn: init feature(H)
        adj: A
        '''
        # support = torch.spmm(infeatn, self.weight)  # H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        # output = torch.spmm(adj, support)  # A*H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        support =torch.matmul(infeatn, self.weight) 
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_layers, activation, dropout):
        super(GCN2, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConvolution2(nfeat, nhid, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution2(nhid, nhid, activation=activation))
        # output layer
        self.layers.append(GraphConvolution2(nhid, nclass))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, adj):

        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, adj)
        return h














