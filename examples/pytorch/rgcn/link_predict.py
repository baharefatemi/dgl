"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from layers import RGCNBlockLayer as RGCNLayer
from layers import RGCNBlockLayer2 as RGCNLayer2

from model import BaseRGCN

import utils
import os
import warnings 
from os import listdir
from os.path import isfile, join
import os.path

# hyperparameter tuning library of Element AI
use_shuriken = True

if(use_shuriken):
    from shuriken.callbacks import ShurikenMonitor
    from shuriken.utils import get_hparams

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, weight, incidence_in, incidence_out):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = self.embedding(node_id)
        return weight


class RGCN0(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=act, self_loop=True, dropout=self.dropout)

class RGCN1(BaseRGCN):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, skip_connection=False, rel_activation=1, rel_dropout=0):
        super(RGCN1, self).__init__(num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers, dropout, use_cuda, skip_connection, rel_activation, rel_dropout)

    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None

        if(idx < self.num_hidden_layers - 1):
            if(self.rel_activation == 0):
                rel_act = nn.Tanh()
            elif(self.rel_activation == 1):
                rel_act = nn.ELU()
            elif(self.rel_activation == 2):
                rel_act = nn.CELU()
            elif(self.rel_activation == 3):
                rel_act= nn.SELU()
            elif(self.rel_activation == 4):
                rel_act = nn.Hardtanh()
            elif(self.rel_activation == 5):
                rel_act = nn.ReLU()
            else:
                rel_act = None
        else:
            rel_act = None

        return RGCNLayer2(self.h_dim, self.h_dim, self.num_rels,
                         activation=act, self_loop=True, dropout=self.dropout, rel_activation=rel_act, rel_dropout=self.rel_dropout)


class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, model_name, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, skip_connection=False, rel_activation=1, rel_dropout=0):
        super(LinkPredict, self).__init__()
        self.model_name = model_name
        if(model_name == "RGCN"):
            self.rgcn = RGCN0(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                             num_hidden_layers, dropout, use_cuda)
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
            nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

        elif(model_name == "EGCN"):
            self.rgcn = RGCN1(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                             num_hidden_layers, dropout, use_cuda, skip_connection, rel_activation, rel_dropout)

        self.reg_param = reg_param

        
    def calc_score(self, embedding, weight, triplets):

        s = embedding[triplets[:,0]]
        if(self.model_name == "RGCN"):
            r = self.w_relation[triplets[:,1]]
        elif(self.model_name == "EGCN"):
            r = weight[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, weight, incidence_in, incidence_out):
        return self.rgcn.forward(g, weight, incidence_in, incidence_out)


    def evaluate(self, g, weight, incidence_in, incidence_out):
        # get embedding and relation weight without grad
        embedding, weight = self.forward(g, weight.cpu(), incidence_in.cpu(), incidence_out.cpu())

        if(self.model_name == "RGCN"):
            return embedding, self.w_relation
        elif(self.model_name == "EGCN"):
            return embedding, weight

    def regularization_loss(self, embedding, weight):
        if(self.model_name == "RGCN"):
            return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))
        if(self.model_name == "EGCN"):
            return torch.mean(embedding.pow(2)) + torch.mean(weight.pow(2))

    def get_loss(self, g, triplets, labels, weight, incidence_in, incidence_out):
        embedding, weight = self.forward(g, weight, incidence_in, incidence_out)
        score = self.calc_score(embedding, weight, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding, weight)
        return predict_loss + self.reg_param * reg_loss, weight


def encode_hype(args):
    encoded = ""
    encoded += str(args.lr) + "_"
    encoded += str(args.dropout) + "_" 
    encoded += str(args.regularization) + "_"
    encoded += str(args.rel_dropout)
    return encoded

# This function is to save the model and optimizer. 
# This helps running jobs restartable
def save_model(args, model, itr, opt):  
    print("Saving the model")
    directory = "models/" + args.model + "/" + args.dataset + "/" + str(args.rel_activation) + "/"
    directory_opt = "optimizers/" + args.model + "/" + args.dataset + "/" + str(args.rel_activation) + "/"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_opt):
        os.makedirs(directory_opt)
    torch.save(model, directory + encode_hype(args) + str(itr) + ".chkpnt")
    torch.save(opt, directory_opt + encode_hype(args) + ".chkpnt")

# Save the best model and its mrr 
def save_best_model(args, model, mrr):  
    print("Saving the best model")
    directory = "models/" + args.model + "/" + args.dataset + "/" + str(args.rel_activation) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    mrr_file = open(directory + encode_hype(args) + "best_mrr.txt", 'w')
    mrr_file.write(str(mrr) + "\n")
    torch.save(model, directory + encode_hype(args) + "best_model.chkpnt")

# In case a job is restarted, load the best model and its mrr
def load_best_model(args, model):
    print("Loading the best model")
    directory = "models/" + args.model + "/" + args.dataset + "/" + str(args.rel_activation) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = encode_hype(args) + "best_model.chkpnt"
    if(isfile(join(directory, f))):
        model = torch.load(directory + f)
        mrr_file = open(directory + encode_hype(args) + "best_mrr.txt", 'r')
        best_mrr = float(mrr_file.read().strip())
    else:
        best_mrr = 0
    print("Best model loaded with mrr: " + str(best_mrr))
    return model, best_mrr

# In case a job is retsrated, load the last model saved
# Load the optimizer so we can start from where it restarted
def load_model(args, model):
    print("Loading the model")
    directory = "models/" + args.model + "/" + args.dataset + "/" + str(args.rel_activation) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    hype_encoding = encode_hype(args)
    related_files = {}
    for f in onlyfiles:
        if f.startswith(hype_encoding):
            itr = f.replace('.chkpnt','')
            itr = itr.replace(hype_encoding,'')
            if itr.isdigit():
                related_files[int(itr)] = f
    if(len(related_files) > 0):
        key = max(related_files)
        model = torch.load(directory + related_files[key])
    else:
        key = 0
    print("Model loaded from itr: " + str(key))
    return model, key


def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels
    all_data = np.concatenate((train_data, valid_data, test_data), axis=0)

    if(use_shuriken):
        monitor = ShurikenMonitor()

    use_cuda = torch.cuda.is_available()

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        args.model,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization,
                        skip_connection=args.skip_connection,
                        rel_activation=args.rel_activation,
                        rel_dropout=args.rel_dropout)

    # check if there is a model with the same hyperparameters saved
    new_model, res = load_model(args, model)
    epoch = 0

    if(res != 0):
        model = new_model
        epoch = res

    best_model, best_mrr = load_best_model(args, model)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    # all_data = torch.LongTensor(all_data.astype(set))


    # build test graph
    test_graph, test_rel, test_norm, incidence_in_test, incidence_out_test = utils.build_test_graph(num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)
    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    test_graph.edata['type'] = test_rel

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # model_state_file = 'model_state_' + str(args.dropout) + '_' + str(args.lr) + '_' + str(args.regularization) + '.pth'
    forward_time = []
    backward_time = []

    print("start training...")

    

    weight = torch.randn(num_rels * 2, args.n_hidden).cuda()

    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels, incidence_in, incidence_out = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1)
        edge_type = torch.from_numpy(edge_type)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
        g.ndata.update({'id': node_id, 'norm': node_norm})
        g.edata['type'] = edge_type

        t0 = time.time()
        loss, weight = model.get_loss(g, data, labels, weight, incidence_in, incidence_out)
        weight = weight.detach()
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)

        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            save_model(args, model, epoch, optimizer)
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")

            # mrr_f = utils.evaluate_filtered(test_graph, model, valid_data, all_data, num_nodes, weight, incidence_in_test, incidence_out_test, hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            mrr = utils.evaluate(test_graph, model, valid_data, num_nodes, weight, incidence_in_test, incidence_out_test, hits=[1, 3, 10], eval_bz=args.eval_batch_size)

            if(use_shuriken):
                monitor.send_info(epoch, {"mrr": mrr})            

            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                best_model = model
                try:
                    os.makedirs(args.model)
                except FileExistsError:
                    pass

                save_best_model(args, model, best_mrr)
            if use_cuda:
                model.cuda()


    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    # checkpoint = torch.load(args.model + "/" + model_state_file)
    if use_cuda:
        best_model.cpu() # test on CPU
    best_model.eval()

    test_mrr = utils.evaluate(test_graph, best_model, test_data, num_nodes, weight, incidence_in_test, incidence_out_test, hits=[1, 3, 10], eval_bz=args.eval_batch_size)

    if(use_shuriken):
        monitor.send_info(epoch, {"test mrr": test_mrr})  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=10,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=100,
            help="perform evaluation every n epochs")
    parser.add_argument("--rgcn")
    parser.add_argument("--model", type=str,
            help="which model do you want to train on?")
    parser.add_argument("--skip-connection", type=bool, default=False,
            help="skip connection in EGCN or not")
    parser.add_argument("--rel-activation", type=int, default=1,
            help="type of activation function for relation aggregation")
    parser.add_argument("--rel-dropout", type=float, default=0.2,
            help="relation dropout probability")

    args = parser.parse_args()
    
    # If using the hyperparameter tuning tool of Element AI
    if(use_shuriken):
        d_params = vars(args)
        # get the hyperparameters from the services
        # returns a dict of hyperparams
        hparams = get_hparams()
        if 'n_iterations' in hparams:
            hparams['number_of_steps'] = hparams['n_iterations'] * 100
        d_params.update(hparams)


    main(args)

