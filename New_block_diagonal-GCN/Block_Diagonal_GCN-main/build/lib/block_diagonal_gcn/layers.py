import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, partitions, layer_type, mode):
        super(GraphConvolution, self).__init__()
        # Input dimension
        self.in_features = in_features
        # Output dimension
        self.out_features = out_features
        # The list of partition indexes, Example: Three partitions [[0,3,7],[1,6,8],[2,4,7,9]]
        self.partitions = partitions

        # "BD_layer_1": First block diagonal layer
        # "Fully_connect": Fully connected layer
        self.layer_type = layer_type

        # Made for BD_layer_1
        # Mode 1: "block_multiplication" for block features
        # Mode 2: "iterative_multiplication" features to save space
        self.mode = mode

        if self.layer_type is "Fully_connect":
            # Fully connected layer, set the weight matrix to (in_features x out_features)
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            # Initialize the weight
            self.reset_parameters()
        else:
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
                if self.mode is "iterative_multiplication":
                    # Use concat instead of block_diag
                    W = torch.cat((W, W_1))
                else:
                    W = torch.block_diag(W,W_1)

        # Set it as torch nn parameter for back propagation
        if self.mode is "iterative_multiplication":
            # Remove all zeros from block weights to save space
            W = Parameter(W.reshape((-1,)))
        else:
            W = Parameter(W)
        return W


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #if self.bias is not None:
            #self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.layer_type is "Fully_connect":
            # Fully connected layer
            #print(input.shape) # 10 x 3
            #print(self.weight.shape) # 3 x 16
            support = torch.mm(input, self.weight)
            #print(support.shape) # 10 x 16
            #print(adj.shape) # 10 x 10
            output = torch.spmm(adj, support)
            #print(output.shape) # 10 x 16
        elif self.layer_type is "BD_layer_1":
            # Block diagonal layer
            if self.mode is "block_multiplication":
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
            elif self.mode is "iterative_multiplication":
                # Slice the non-zero block_weight matrix into len(self.partitions) weights,
                # each weight matrix has (self.in_features x self.out_features)
                weights = self.block_weight.view(len(self.partitions), self.in_features, -1)
                # Matrix multiplication iteratively to save space from block features
                S = torch.mm(input[self.partitions[0]], weights[0])
                for i in range(len(self.partitions)):
                    if i is not 0:
                        S = torch.block_diag(S,torch.mm(input[self.partitions[i]], weights[i]))
                output = torch.spmm(adj, S)

        else:
            # The second Block diagonal layer
            support = torch.mm(input, self.block_weight)
            output = torch.spmm(adj, support)

        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
