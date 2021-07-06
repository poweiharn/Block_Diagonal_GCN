import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, partitions, is_block_layer, is_first_layer):
        super(GraphConvolution, self).__init__()
        # Input dimension
        self.in_features = in_features
        # Output dimension
        self.out_features = out_features
        # The list of partition indexes, Example: Three partitions [[0,3,7],[1,6,8],[2,4,7,9]]
        self.partitions = partitions
        # Is this the first block diagonal layer
        self.is_first_layer = is_first_layer
        # Is this layer a block diagonal layer
        self.is_block_layer = is_block_layer

        if not is_block_layer:
            # Fully connected layer, set the weight matrix to (in_features x out_features)
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            # Initialize the weight
            self.reset_parameters()
        else:
            # Block diagonal weight matrix
            self.block_weight = self.initialize_weight()




    def initialize_weight(self):
        # Initialize the weight
        W = torch.FloatTensor(self.in_features, self.out_features)
        stdv = 1. / math.sqrt(W.size(1))
        W.data.uniform_(-stdv, stdv)
        # Iterate through the number of partitions
        for i in range(len(self.partitions)):
            if i is not 0:
                # Initialize the weight
                W_1 = torch.FloatTensor(self.in_features, self.out_features)
                stdv = 1. / math.sqrt(W_1.size(1))
                W_1.data.uniform_(-stdv, stdv)
                # Construct the Block diagonal weight matrix
                W = torch.block_diag(W,W_1)
        # Set it as torch nn parameter for back propagation
        W = Parameter(W)
        return W


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #if self.bias is not None:
            #self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if not self.is_block_layer:
            # Fully connected layer
            #print(input.shape) # 10 x 3
            #print(self.weight.shape) # 3 x 16
            support = torch.mm(input, self.weight)
            #print(support.shape) # 10 x 16
            #print(adj.shape) # 10 x 10
            output = torch.spmm(adj, support)
            #print(output.shape) # 10 x 16
        else:
            # Block diagonal layer
            if self.is_first_layer:
                # If this Block diagonal layer is the first layer
                # Initialize the hidden block features with respect to the partition indices
                X = input[self.partitions[0]]
                for part in self.partitions:
                    if self.partitions.index(part) is not 0:
                        X = torch.block_diag(X,input[part])
                #print(X)
                #print(X.shape) # 10 x 9

                # Matrix Multiplication
                support = torch.mm(X, self.block_weight)
                output = torch.spmm(adj, support)
            else:
                # The second Block diagonal layer
                support = torch.mm(input, self.block_weight)
                output = torch.spmm(adj, support)

        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
