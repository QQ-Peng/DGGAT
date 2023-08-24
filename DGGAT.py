import  os, json, pickle
from models.definitions.GAT_bak import GAT

from utils.constants import *
import utils.utils as utils


import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, VGAE

from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops



class DGGAT(nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    gate_heads = 2
    def __init__(self,input_dim,gat1,temperature=0.2):
        super(DGGAT, self).__init__()
        self.gat1 = gat1
        self.gcn1 = ChebConv(input_dim,300,K=2,normalization="sym")
        self.skip = nn.Linear(input_dim, 300)
        self.linear_proj_gate = nn.Linear(input_dim, self.gate_heads * 100)
        self.scoring_fn_target_gate = nn.Parameter(torch.Tensor(1, self.gate_heads, 100))
        self.scoring_fn_source_gate = nn.Parameter(torch.Tensor(1, self.gate_heads, 100))
        self.temperature = temperature
        self.init_params_gate()
        
    def init_params_gate(self):
        nn.init.xavier_uniform_(self.scoring_fn_target_gate)
        nn.init.xavier_uniform_(self.scoring_fn_source_gate)
    def lift(self, source_scores_gate, target_scores_gate, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        source_scores_gate = source_scores_gate.index_select(self.nodes_dim, src_nodes_index)
        target_scores_gate = target_scores_gate.index_select(self.nodes_dim, trg_nodes_index)
        return source_scores_gate, target_scores_gate
    def forward(self, data, true_y, mask,drop_method='gate'):
        # print(drop_method)
        edge_index = data.edge_index
        if drop_method == 'gate':
            nodes_features_proj = self.linear_proj_gate(data.x).view(-1, self.gate_heads, 100)
            # (N,NH,FOUT)*(1,NH,FOUT)
            scores_source_gate = (nodes_features_proj * self.scoring_fn_source_gate).sum(dim=-1)
            scores_target_gate = (nodes_features_proj * self.scoring_fn_target_gate).sum(dim=-1)
            scores_source_gate, scores_target_gate = self.lift(scores_source_gate, scores_target_gate, edge_index)
            scores_per_edge_gate = (scores_source_gate + scores_target_gate).mean(-1).sigmoid().unsqueeze(-1)
            scores_per_edge_gate = torch.cat([scores_per_edge_gate, 1 - scores_per_edge_gate], -1)
            #scores_per_edge_gate = self.gumbel_softmax(logits=scores_per_edge_gate, hard=True)[:, 0].bool()
            scores_per_edge_gate = self.gumbel_softmax(logits=scores_per_edge_gate, hard=True)[:, 0]
        else:
            scores_per_edge_gate = None
        if drop_method == 'gate':
            edge_index = edge_index.t()[scores_per_edge_gate.bool()].t()
        elif drop_method == 'random':
            edge_index, _ = dropout_adj(edge_index, p=0.5,
                                        force_undirected=True,
                                        num_nodes=data.x.size()[0],
                                        training=self.training)
        else:
            pass  # no drop
        x = data.x
        x_skip = F.dropout(F.relu(self.skip(data.x)),training=self.training)
        x_global = F.dropout(F.relu(self.gcn1(x, edge_index)),training=self.training)

        if drop_method == 'gate':
            pred = self.gat1((x_global + x_skip, data.edge_index,scores_per_edge_gate.unsqueeze(-1)))[0]
        else:
            pred = self.gat1((x_global + x_skip, edge_index, scores_per_edge_gate))[0]
        pred_loss = F.binary_cross_entropy_with_logits(pred[mask], true_y[mask])
        if drop_method == 'gate':
            pred_loss = pred_loss + 2*scores_per_edge_gate.sum()/scores_per_edge_gate.shape[0]

        return edge_index, pred_loss, pred
    def sample_gumbel(self, shape, eps=1e-15, device='cuda'):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), device=logits.device)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, self.temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard
    

class VariationalGAE(torch.nn.Module):
    def __init__(self,node_num, in_channels, out_channels):
        super(VariationalGAE, self).__init__()
        self.trans = nn.Linear(node_num,in_channels)
        self.conv1 = ChebConv(in_channels, 2 * out_channels,K=2,normalization='sym') 
        self.conv_mu = ChebConv(2 * out_channels, out_channels,K=2,normalization='sym')
        self.conv_logstd = ChebConv(2 * out_channels, out_channels,K=2,normalization='sym')

    def forward(self, x, edge_index):
        x = self.trans(x).relu()
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGGNN(nn.Module):
    def __init__(self,node_num,hidden_dim=128,z_dim=32) -> None:
        super().__init__()
        self.VGAE = VGAE(VariationalGAE(node_num,hidden_dim,z_dim))
        self.node_num = node_num
    def forward(self,x_adj, edge_index,neg_edge_index=None):
        z = self.VGAE.encode(x_adj, edge_index)
        loss = self.VGAE.recon_loss(z, edge_index,neg_edge_index)
        loss = loss + (1 / self.node_num) * self.VGAE.kl_loss()
        return z,loss
