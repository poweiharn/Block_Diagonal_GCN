import numpy as np
import scipy.sparse as sp

path="../data/test/"
dataset="test"

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = encode_onehot(idx_features_labels[:, -1])

idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)

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

def get_adjacency_matrix(edge_list, node_index):
    adj = np.zeros((len(node_index), len(node_index)))
    for index, value in enumerate(node_index):
        for index1, value1 in enumerate(node_index):
            for edge in edge_list:
                if edge[0] == value and edge[1] == value1:
                    adj[index][index1] = 1

    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


idx1 = [0,3,7]
adj1 = get_adjacency_matrix(edges,idx1)
print(normalize(adj1 + sp.eye(adj1.shape[0])).toarray())
adj_normalized1 = normalize_adj(adj1 + sp.eye(adj1.shape[0]))
print(adj_normalized1.toarray())

idx2 = [1,6,8]
adj2 = get_adjacency_matrix(edges,idx2)
print(adj2.toarray())
adj_normalized2 = normalize_adj(adj2 + sp.eye(adj2.shape[0]))
print(adj_normalized2.toarray())

idx3 = [2,4,5,9]
adj3 = get_adjacency_matrix(edges,idx3)
print(adj3.toarray())
adj_normalized3 = normalize_adj(adj3 + sp.eye(adj3.shape[0]))
print(adj_normalized3.toarray())


idx_train = idx1 + idx2 + idx3
adj_block = sp.block_diag([adj1, adj2, adj3])
adj_normalized_block = normalize_adj(adj_block + sp.eye(adj_block.shape[0]))
print(adj_normalized_block.toarray())


print(idx_train)
print(adj_block.toarray())
print(features.toarray())
print(features[idx1].toarray())
print(features[idx_train].toarray())


