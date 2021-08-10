import torch.nn as nn
import torch.nn.functional as F
from block_diagonal_gcn.layers import GraphConvolution


class GCN_B_D(nn.Module):
    def __init__(self, nfeatures, nclass, partitions, dropout):
        super(GCN_B_D, self).__init__()
        self.partition_num = len(partitions)
        self.layer1_hidden = 16
        self.layer2_hidden = 16

        # Block diagonal layer 1
        self.gc1 = GraphConvolution(nfeatures, self.layer1_hidden, partitions, "BD_layer_1", "iterative_multiplication")
        # Block diagonal layer 2
        self.gc2 = GraphConvolution(self.layer1_hidden, self.layer2_hidden, partitions, "BD_layer_2", "block_multiplication")
        # Fully connected layer
        self.gc3 = GraphConvolution(self.layer2_hidden*self.partition_num, nclass, partitions, "Fully_connect", False)
        self.dropout = dropout

    def forward(self, x, adj):
        #print("--------gc1-----------")
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        #print(self.gc1.block_weight)

        #print("--------gc2-----------")
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # print(self.gc2.block_weight)

        #print("--------gc3-----------")
        x = self.gc3(x, adj)
        # print(self.gc3.weight)
        return F.log_softmax(x, dim=1)
