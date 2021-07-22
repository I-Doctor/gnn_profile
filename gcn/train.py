import argparse
import time
import numpy as np
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch
import torch.nn.functional as F

from gcn import GCN
from gcn_mp import GCN as GCNmp
#from gcn_spmv import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):

    # load and preprocess dataset
    tic = time.time()
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]
    toc = time.time()
    print('PROFILING dataset time: {}'.format(toc-tic))

    if args.gpu < 0:
        cuda = False
    else:
        tic = time.time()
        cuda = True
        g = g.int().to(args.gpu)
        toc = time.time()
        print('PROFILING to gpu time: {}'.format(toc-tic))

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # preprocess
    tic = time.time()
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    toc = time.time()
    print('PROFILING preprocess time: {}'.format(toc-tic))

    # create GCN model
    tic = time.time()
    if args.mp:
        model = GCNmp(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)
    else:
        model = GCN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    toc = time.time()
    print('PROFILING model time: {}'.format(toc-tic))

    # initialize graph
    dur = []
    PROF_forw = []
    PROF_loss = []
    PROF_back = []
    for epoch in range(args.n_epochs):

        model.train()
        t0 = time.time()

        tic = time.time()
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False) as prof:
            logits = model(features)
        if epoch == 5:
            print(prof.table())
            exit()
        PROF_forw.append(time.time() - tic)

        tic = time.time()
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        PROF_loss.append(time.time() - tic)

        tic = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        PROF_back.append(time.time() - tic)

        dur.append(time.time() - t0)
        acc = evaluate(model, features, labels, val_mask)
        gpu_mem = torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
        print("Epoch {:05d} | Time(s) epoch/forw/loss/back {:.5f}/{:.5f}/{:.5f}/{:.5f} | " 
                "Loss {:.4f} | Accuracy {:.4f} | ETputs(KTEPS) {:.2f} | GPU Mem (MB) {:.2f}".format(epoch, 
                np.mean(dur[3:]), np.mean(PROF_forw[3:]), np.mean(PROF_loss[3:]), 
                np.mean(PROF_back[3:]), loss.item(), acc, n_edges/np.mean(dur[3:])/1000, gpu_mem))

    tic = time.time()
    acc = evaluate(model, features, labels, test_mask)
    toc = time.time()
    print('PROFILING test evaluate time: {}'.format(toc-tic))
    print("Test accuracy {:.4%}".format(acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--mp", action='store_true',
                        help="use gcn_mp.py which is user-defined message and reduce functions")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)

