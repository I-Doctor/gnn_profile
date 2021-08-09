import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args, load_data
from models import *
from conf import *
import networkx as nx

from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import time

def get_model_and_config(name):
    name = name.lower()
    if name == 'gcn':
        return GCN, GCN_CONFIG
    elif name == 'gat':
        return GAT, GAT_CONFIG
    elif name == 'graphsage':
        return GraphSAGE, GRAPHSAGE_CONFIG
    elif name == 'appnp':
        return APPNP, APPNP_CONFIG
    elif name == 'tagcn':
        return TAGCN, TAGCN_CONFIG
    elif name == 'agnn':
        return AGNN, AGNN_CONFIG
    elif name == 'sgc':
        return SGC, SGC_CONFIG
    elif name == 'gin':
        return GIN, GIN_CONFIG
    elif name == 'chebnet':
        return ChebNet, CHEBNET_CONFIG

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
    begin_load_data_time = datetime.now()
    data = load_data(args)
    end_load_data_time = datetime.now()
    print('Load Data in: '+str((end_load_data_time-begin_load_data_time).total_seconds()))
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)
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
    begin_pre_process_data_time = datetime.now()
    # graph preprocess and calculate normalization factor
    # add self loop
    if args.self_loop:
        g = g.remove_self_loop().add_self_loop()
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    end_pre_process_data_time = datetime.now()
    print('Pre-process Data in: '+str((end_pre_process_data_time-begin_pre_process_data_time).total_seconds()))

    # create GCN model
    GNN, config = get_model_and_config(args.model)
    model = GNN(g,
                in_feats,
                n_classes,
                *config['extra_args'])

    if cuda:
        model = model.cuda()

    print(model)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    total_forward_time = 0
    total_backward_time = 0
    dur = []
    for epoch in range(200):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        begin_forward_time = time.time()
        with record_function('forward'):
            logits = model(features)
        end_forward_time = time.time()
        total_forward_time += (end_forward_time - begin_forward_time)

        loss = loss_fcn(logits[train_mask], labels[train_mask])
        begin_backward_time = time.time()
        with record_function('backward'):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_backward_time = time.time()
        total_backward_time += (end_backward_time - begin_backward_time)

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    print('Avg Forward Time: ' + str(total_forward_time/200))
    print('Avg Backward Time: ' + str(total_backward_time/200))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification on citation networks.')
    register_data_args(parser)
    parser.add_argument("--model", type=str, default='gcn',
                        help='model to use, available models are gcn, gat, graphsage, gin,'
                             'appnp, tagcn, sgc, agnn')
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    args = parser.parse_args()
    print(args)
    # main(args)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True, record_shapes=True, use_cuda=True) as p:
        main(args)
    p.export_chrome_trace('profile_model-zoo-gat-citeseer.json')
    print(p.key_averages().table(sort_by="cuda_time_total"))
    #print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # old version，not right
    # print("Total CPU Time (microseconds):")
    # print(sum([item.cpu_time for item in p.function_events]))
    # print("Total CUDA Time (microseconds):")
    # print(sum([item.cuda_time for item in p.function_events]))
