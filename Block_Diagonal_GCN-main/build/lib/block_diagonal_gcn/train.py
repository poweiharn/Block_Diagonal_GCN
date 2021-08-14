from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim

from block_diagonal_gcn.utils import load_data, process_data, accuracy, plot_confusion, plot_tsne
from block_diagonal_gcn.models import GCN_B_D
from block_diagonal_gcn.weight_pruning import ThresholdPruning

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--hidden', type=int, default=16,
                    #help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_pruning_threshold', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, partitions, idx_train, idx_val, idx_test = load_data()
adj, features, labels, partitions, idx_train, idx_val, idx_test = process_data("./data/","citeseer")

# Model and optimizer
print('labels', labels.max().item() + 1)
model = GCN_B_D(nfeatures=features.shape[1],
            nclass=labels.max().item() + 1,
            partitions=partitions,
            dropout=args.dropout)

#for param in model.parameters():
    # Weights for back propagation
    #print(type(param), param.size())

#parameters_to_prune = (
#    (model.gc1, 'block_weight'),
#    #(model.gc2, 'block_weight'),
#    (model.gc3, 'weight'),
#)


#prune.global_unstructured(
#    parameters_to_prune,
#    pruning_method=ThresholdPruning, threshold=args.weight_pruning_threshold,
#)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    '''print('adj dim', adj.shape)
    print('test indices', idx_test)
    print('outpt for last index', output[2432])
    print('size', output.size())'''
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    '''print(
        "Sparsity in gc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc1.block_weight == 0))
            / float(model.gc1.block_weight.nelement())
        )
    )
    print(
        "Sparsity in gc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc2.block_weight == 0))
            / float(model.gc2.block_weight.nelement())
        )
    )
    print(
        "Sparsity in gc3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc3.weight == 0))
            / float(model.gc3.weight.nelement())
        )
    )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                + torch.sum(model.gc1.block_weight == 0)
                + torch.sum(model.gc2.block_weight == 0)
                + torch.sum(model.gc3.weight == 0)
            )
            / float(
                + model.gc1.block_weight.nelement()
                + model.gc2.block_weight.nelement()
                + model.gc3.weight.nelement()
            )
        )
    )'''


def test():
    model.eval()
    output = model(features, adj)
    #print(labels[idx_test])
    #print(output[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    #plot_confusion(output[idx_test], labels[idx_test])
    #plot_tsne(output[idx_test], labels[idx_test], features[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
