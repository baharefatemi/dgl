"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import numpy as np
import torch
import dgl
import math
from dgl.contrib.data import load_data

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """ Edge neighborhood sampling to reduce training graph size
    """

    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm, incidence_in, incidence_out = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels, incidence_in, incidence_out

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))

    incidence_in = torch.zeros(num_rels * 2, num_nodes).cuda()
    incidence_out = torch.zeros(num_rels * 2, num_nodes).cuda()

    # print(edges)
    for edge in edges:
        incidence_in[edge[2], edge[0]] = 1

    for edge in edges:
        incidence_out[edge[2], edge[1]] = 1

    return g, rel, norm, incidence_in, incidence_out

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility function for evaluations
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    # print(target)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(embedding, w, a, r, b, num_entity, num_triples, batch_size):
    """ Perturb one element in the triplets
    """
    # n_batch = (num_entity + batch_size - 1) // batch_size
    n_batch = math.floor(num_triples // batch_size)

    ranks = []
    for idx in range(n_batch):
        # print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(num_entity, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        
        emb_ar = emb_ar.cuda()
        emb_c = emb_c.cuda()

        # print(idx)

        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score).cpu()
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# TODO (lingfan): implement filtered metrics
# return MRR (raw), and Hits @ (1, 3, 10)
def evaluate(test_graph, model, test_triplets, num_entity, weight, incidence_in_test, incidence_out_test, hits=[], eval_bz=10):

    with torch.no_grad():
        embedding, w = model.evaluate(test_graph, weight, incidence_in_test, incidence_out_test)

        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, len(test_triplets), eval_bz)

        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, len(test_triplets), eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()



def evaluate_filtered(test_graph, model, test_triplets, all_data, num_entity, weight, incidence_in_test, incidence_out_test, hits=[], eval_bz=10):

    with torch.no_grad():
        embedding, w = model.evaluate(test_graph, weight, incidence_in_test, incidence_out_test)

        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        # perturb subject
        ranks_s = perturb_and_get_rank_filtered(embedding, w, o, r, s, num_entity, len(test_triplets), all_data, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank_filtered(embedding, w, s, r, o, num_entity, len(test_triplets), all_data, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()


def perturb_and_get_rank_filtered(embedding, w, a, r, b, num_entity, num_triples, all_data, batch_size):
    """ Perturb one element in the triplets
    """
    # n_batch = (num_entity + batch_size - 1) // batch_size
    # n_batch = math.floor(num_triples // batch_size)

    ranks = []
    for idx in range(num_triples):
        a_r = torch.tensor([a[idx], r[idx]]).repeat(num_entity, 1)
        b = torch.arange(num_entity).view(-1, 1)
        
        triples = torch.cat((a_r, b), dim=1).numpy()
        to_test_triples = np.setdiff1d(triples, all_data)

        print(len(triples))
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1

        # I should change emb_c

        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        
        emb_ar = emb_ar.cuda()
        emb_c = emb_c.cuda()

        # print(idx)

        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score).cpu()
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)



# def perturb_and_get_rank_filtered(embedding, w, s, r, o, num_entity, all_data):
#     """ Perturb one element in the triplets
#     """

#     # embedding stores node embedding
#     # w stores relation embedding


#     casted_all_data = cast_all_data(all_data)

#     rank_s = 0
#     rank_o = 0
#     for head_or_tail in ["head", "tail"]:
#         queries = create_queries([s, r, o], head_or_tail, num_entity)
#         # print(tuple([s, r, o]))
#         # print(queries)
#         # print(set(queries))
#         # all_data = all_data.astype(set)
#         # print(all_data)
#         # print(set(queries))
#         # a = list(set(queries) - all_data)
#         # queries = [tuple([s, r, o])] + a
        
#         # print(set(queries))

#         queries = list(set(queries) - casted_all_data)

#         if(head_or_tail == "head"):
#             a = o
#             b = queries[:, 0]
#         else:
#             a = s
#             b = queries[:, 2]

#         emb_ar = embedding[a] * w[r]
#         emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
#         emb_c = embedding[b].transpose(0, 1).unsqueeze(1) # size: D x 1 x V

#         # out-prod and reduce sum
#         out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
#         score = torch.sum(out_prod, dim=0) # size E x V
#         score = torch.sigmoid(score)
#         target = b[0]
#         rank = sort_and_rank(score, target)
#         if(head_or_tail == "head"):
#             rank_s = rank
#         else:
#             rank_o = rank

#     return rank_s, rank_o

# def create_queries(fact, head_or_tail, num_entity):
#     head, rel, tail = fact
#     if head_or_tail == "head":
#         return [(torch.tensor(i), rel, tail) for i in range(num_entity)]
#     elif head_or_tail == "tail":
#         return [(head, rel, torch.tensor(i)) for i in range(num_entity)]


# return MRR (filtered), and Hits @ (1, 3, 10)
# def evaluate_filtered(test_graph, model, test_triplets, num_entity, all_data, hits=[]):
#     ranks = []
#     with torch.no_grad():
#         embedding, w = model.evaluate(test_graph)
#         for i in range(len(test_triplets)):
#             s = test_triplets[i, 0]
#             r = test_triplets[i, 1]
#             o = test_triplets[i, 2]

#             # perturb subject and object
#             rank_s, rank_o = perturb_and_get_rank_filtered(embedding, w, o, r, s, num_entity, all_data)

#             ranks.append(rank_s)
#             ranks.append(rank_o)
#         # ranks += 1 # change to 1-indexed
#         ranks = torch.tensor(ranks)
#         mrr = torch.mean(1.0 / ranks.float())
#         print("MRR (filtered): {:.6f}".format(mrr.item()))

#         for hit in hits:
#             avg_count = torch.mean((ranks <= hit).float())
#             print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
#     return mrr.item()

# def cast_all_data(all_data):
#     casted_data = set()
#     for i in all_data:
#         s = torch.tensor(i[0])
#         r = torch.tensor(i[1])
#         o = torch.tensor(i[2])
#         # use_cuda = torch.cuda.is_available()
#         # if(use_cuda):
#         #     s = s.cuda()
#         #     r = r.cuda()
#         #     o = o.cuda()
#         triple = tuple([s, r, o])
#         casted_data.add(triple)
#     return casted_data

