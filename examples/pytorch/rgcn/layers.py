import torch
import torch.nn as nn
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g, weight, incidence_in, incidence_out):
        raise NotImplementedError

    def forward(self, g, weight, incidence_in, incidence_out):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        weight_ = self.propagate(g, weight, incidence_in, incidence_out)


        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr

        return weight_

class RGCNLayer2(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0, skip_connection=False):
        super(RGCNLayer2, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connection = skip_connection

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g, weight, incidence_in, incidence_out):
        raise NotImplementedError

    def forward(self, g, weight, incidence_in, incidence_out):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        # store previous value
        if self.skip_connection:
            prev_h = g.ndata['h']

        # In the propagate function the data in 'h' will be updated.
        weight_ = self.propagate(g, weight, incidence_in, incidence_out)


        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.skip_connection:
            node_repr = node_repr + prev_h

        g.ndata['h'] = node_repr
        return weight_




class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)

class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g, weight, incidence_in, incidence_out):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        return weight
    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class RGCNBlockLayer2(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connection=False):
        super(RGCNBlockLayer2, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        # , skip_connection=skip_connection)
        # self.skip_connection = skip_connection
        self.num_rels = num_rels
        self.out_feat = out_feat
        

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # print("init weight:")
        # print(len(torch.nonzero(self.weight))/self.weight.numel())


        # assuming in_feat and out_feat are both divisible by num_bases
        # self.weight = torch.rand(self.num_rels, self.out_feat).cuda()
        # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # self.weight.requires_grad = False

        # self.W1 = nn.Linear(out_feat, out_feat, bias=True)
        self.W2 = nn.Linear(out_feat, out_feat, bias=True)
        self.W3 = nn.Linear(out_feat, out_feat, bias=True)
        self.W4 = nn.Linear(out_feat, out_feat, bias=True)
        self.W5 = nn.Linear(out_feat, out_feat, bias=True)
        self.W6 = nn.Linear(out_feat, out_feat, bias=True)

    def msg_func(self, edges):


        # print("msg_function:")
        # print(self.weight)
        # print("weight after relation aggregating:")
        # print(self.weight)

        # print("weight:")
        # print(len(torch.nonzero(self.weight))/self.weight.numel())

        weight1 = self.weight.index_select(0, edges.data['type'])
        # .view(-1, self.out_feat)
        
        node = edges.src['h']
        # node1 = edges.dst['h']

        msg = torch.sigmoid(self.W2(weight1)) * self.W3(node)

        return {'msg': msg}

    def propagate(self, g, weight, incidence_in, incidence_out):

        n_0 = torch.sum(incidence_in.float(), dim = 1).view(-1, 1)
        n_1 = torch.sum(incidence_out.float(), dim = 1).view(-1, 1)

        n_0[n_0 == float(0)] = 0.000000001
        n_1[n_1 == float(0)] = 0.000000001

        incidence_in = incidence_in / n_0
        incidence_out = incidence_out / n_1

        # print("weight before relation aggregation:")
        # print(weight)
        # print(len(torch.nonzero(self.weight))/self.weight.numel())
        # elu = torch.nn.ELU()
        
        self.weight =  nn.Parameter(torch.tanh(self.W4(weight) + self.W5(torch.mm(incidence_in, g.ndata['h'])) + self.W6(torch.mm(incidence_out, g.ndata['h']))))
        # self.weight = nn.Parameter(weight)

        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        
        # print("weight after relation aggregation:")
        # print(self.weight)

        return self.weight

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


