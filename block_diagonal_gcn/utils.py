import numpy as np
import scipy.sparse as sp
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.manifold import TSNE
from block_diagonal_gcn.partition import clique_partition, general_partition 


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_adjacency_matrix(edge_list, node_index):
    adj = np.zeros((len(node_index), len(node_index)))
    for index, value in enumerate(node_index):
        for index1, value1 in enumerate(node_index):
            for edge in edge_list:
                if edge[0] == value and edge[1] == value1:
                    adj[index][index1] = 1

    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj



def load_data(path="../data/test/", dataset="test"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    '''idx1 = [0, 3, 7]
    adj1 = get_adjacency_matrix(edges, idx1)
    print(adj1.toarray())

    idx2 = [1, 6, 8]
    adj2 = get_adjacency_matrix(edges, idx2)
    print(adj2.toarray())

    idx3 = [2, 4, 5, 9]
    adj3 = get_adjacency_matrix(edges, idx3)
    print(adj3.toarray())'''
    
    partition_type='clique partition'
    if(partition_type=='clique partition'):
      adj1, adj2, adj3, idx1, idx2, idx3 = clique_partition(edges)
    else:
      adj1, adj2, adj3, idx1, idx2, idx3 = general_partition(edges)
    print('indices list:')
    print('idx1', idx1)
    print('idx2', idx2)
    print('idx3', idx3)
    print('before normalization:', adj1.toarray())
    adj1 = sp.coo_matrix(adj1, dtype=np.float32)
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj1 = normalize_adj(adj1 + sp.eye(adj1.shape[0]))
    print('after normalization:', adj1.toarray())

    print('before normalization:', adj2.toarray())
    adj2 = sp.coo_matrix(adj2, dtype=np.float32)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    adj2 = normalize_adj(adj2 + sp.eye(adj2.shape[0]))
    print('after normalization:', adj2.toarray())

    print('before normalization:', adj3.toarray())
    adj3 = sp.coo_matrix(adj3, dtype=np.float32)
    adj3 = adj3 + adj3.T.multiply(adj3.T > adj3) - adj3.multiply(adj3.T > adj3)
    adj3 = normalize_adj(adj3 + sp.eye(adj3.shape[0]))
    print('after normalization:', adj3.toarray())
    
    adj_block = sp.block_diag([adj1, adj2, adj3])

    partitions = []
    partitions.append(idx1)
    partitions.append(idx2)
    partitions.append(idx3)

    idx_train = idx1 + idx2 + idx3
    idx_val = range(4, 8)
    idx_test = range(8, 10)

    features = normalize(features)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    #print(adj_block.toarray())
    adj = sparse_mx_to_torch_sparse_tensor(adj_block)
    #print(adj.to_dense())

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, partitions, idx_train, idx_val, idx_test


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # Compute D^{-0.5} * A * D^{-0.5}
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def plot_confusion(output, labels):
    preds = output.max(1)[1].type_as(labels)
    multiclass = np.array(confusion_matrix(labels, preds))
    fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True)
    #plt.savefig("confusion.png")
    plt.show()


def plot_tsne(output, labels, features):
    preds = output.max(1)[1].type_as(labels)
    tsne = TSNE(n_components=2)
    low_dim_embs = tsne.fit_transform(features)
    plt.title('Tsne result')
    plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], marker='o', c=preds, cmap="jet", alpha=0.7, )
    #plt.savefig("tsne.png")
    plt.show()



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
