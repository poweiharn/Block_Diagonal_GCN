import sys
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp

def norm_feat(features):
    row_sum_inv = np.power(np.sum(features, axis=1), -1)
    row_sum_inv[np.isinf(row_sum_inv)] = 0.
    deg_inv = np.diag(row_sum_inv)
    norm_features = np.dot(deg_inv, features)
    norm_features = np.array(norm_features, dtype=np.float32)

    return norm_features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_data(dataset_name, dataset_path):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_path = dataset_path + dataset_name + '/'
    for i in range(len(names)):
        with open(dataset_path + 'ind.{}.{}'.format(dataset_name.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataset_path + 'ind.{}.test.index'.format(dataset_name))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    features = norm_feat(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.csc_matrix(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    return features, labels, adj, idx_train, idx_val, idx_test


def main():
    dataset_name = 'citeseer'
    data_path = './data/'
    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'pubmed':
        features, labels, dir_adj, idx_train, idx_val, idx_test = load_citation_data(dataset_name, data_path)
        print(features)
        print(labels)
        print(dir_adj)
        print(idx_train)
        print(idx_val)
        print(idx_test)


if __name__ == "__main__":
    main()