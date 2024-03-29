import argparse, os, sys, json, time, pickle
sys.path.append("./GAT")
from models.definitions.GAT_bak import GAT

from utils.constants import *
import utils.utils as utils

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import pandas as pd
import networkx as nx

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
from sklearn import metrics
from DGGAT import DGGAT, VGGNN
import argparse
import random



def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# same_seeds(seed)

@torch.no_grad()
def test(model, data, Y, mask):
    model.eval()
    _, pred_loss, pred = model(data, Y, mask)
    pred = torch.sigmoid(pred[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    model.train()
    return metrics.roc_auc_score(Yn, pred), area

def buid_GAT(log_weight=False):
    gat = GAT(num_of_layers=2,  # config['num_of_layers'],
        num_heads_per_layer=[2, 2],  # config['num_heads_per_layer'],
        num_features_per_layer=[300, 100, 1],  # config['num_features_per_layer'],
        add_skip_connection=True,  # config['add_skip_connection'],
        bias=True,  # config['bias'],
        dropout=0.0,  # config['dropout'],
        layer_type=LayerType.IMP3,  # config['layer_type'],
        log_attention_weights=log_weight)
    return gat

def cross_val(EPOCH,data, Y,k_sets,dropmethod,outdir):

    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))

    for i in range(10):
        for cv_run in range(5):
            gat1 = buid_GAT()
            tr_mask, te_mask = k_sets[i][cv_run]
            model = DGGAT(64,gat1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(0, EPOCH):
                optimizer.zero_grad()
                edge_index, pred_loss, pred = model(data, Y, tr_mask,dropmethod)
                pred_loss.backward()
                optimizer.step()
            auc, aupr = test(model, data, Y, te_mask)
            AUC[i][cv_run] = auc
            AUPR[i][cv_run] = aupr
            print(f"i: {i}, cv_run: {cv_run} done.")
    np.save(f'{outdir}/AUC_{dropmethod}.npy',AUC)
    np.save(f'{outdir}/AUPR_{dropmethod}.npy',AUPR)



def train(EPOCH,data, Y,mask_all,dropmethod,outdir):
    same_seeds(6666)
    gat1 = buid_GAT()
    model = DGGAT(64,gat1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    all_loss = []
    for epoch in range(1, EPOCH):
        optimizer.zero_grad()
        edge_index, pred_loss, pred = model(data, Y, mask_all,dropmethod)
        pred_loss.backward()
        optimizer.step()
        all_loss.append(pred_loss.item())
        if epoch > 30:
            r = ((np.mean(all_loss[epoch - 20:epoch - 10]) - np.mean(all_loss[epoch - 10:epoch])) / np.mean(
                all_loss[epoch - 20:epoch - 10]))
            if  r<= 0.01:
                break
    torch.save(model.state_dict(),
               f'{outdir}/model_{dropmethod}.bin')


def predict(data, Y,mask_all,dropmethod,outdir,model_path,data_dir):
    # same_seeds(6666)
    device = Y.device
    weight = torch.load(model_path,map_location=torch.device('cpu'))
    gat1 = buid_GAT(True)
    model = DGGAT(64,gat1).to(device)
    model.load_state_dict(weight)
    model.eval()
    for i in range(100):
        print("*" * 30)
        _, pred_loss, pred = model(data, Y, mask_all,dropmethod)
        edge_index = data.edge_index
        pred = torch.sigmoid(pred).detach().cpu().numpy()
        pred1 = pred

        node_names = data.node_names[:, 1].tolist()
        attention_weights = model.gat1.gat_net[1].attention_weights.squeeze(dim=-1)
        # print(attention_weights.shape)
        target_neighbor_weights = {}
        target_neighbor = {}
        for t in node_names:
            target_nh = edge_index[1] == node_names.index(t)
            edge = edge_index.t()[target_nh]
            attention = attention_weights[target_nh]
            target_neighbor_weights[t] = attention.mean(-1).tolist()
            tmp = []
            for s in edge[:, 0]:
                tmp.append(node_names[s])
            target_neighbor[t] = tmp

        edge_index = edge_index.t()
        attention_weights = attention_weights.mean(-1, keepdim=True)

        # pred = torch.load(pred_path)
        node_names = data.node_names[:, 1].tolist()
        oncokb = pd.read_csv(f"{data_dir}/OncoKB_cancerGeneList.tsv", sep='\t')[
            'Hugo Symbol'].tolist()
        ongene = pd.read_csv(f"{data_dir}/ongene_human.txt", sep='\t')[
            'OncogeneName'].tolist()
        test_samples = data.node_names[~mask_all][:, 1].tolist()
        y_oncokb_independent = [i in oncokb for i in test_samples]
        y_ongene_independent = [i in ongene for i in test_samples]
        # print(pred.shape)
        precision, recall, _thresholds = metrics.precision_recall_curve(y_oncokb_independent, pred[~mask_all])
        aupr_oncokb = metrics.auc(recall, precision)
        auc_oncokb = metrics.roc_auc_score(y_oncokb_independent, pred[~mask_all])
        print("oncokb: ",auc_oncokb, "   ", aupr_oncokb)
        precision, recall, _thresholds = metrics.precision_recall_curve(y_ongene_independent, pred[~mask_all])
        aupr_ongene = metrics.auc(recall, precision)
        auc_ongene = metrics.roc_auc_score(y_ongene_independent, pred[~mask_all])
        print("ongene",auc_ongene , "   ", aupr_ongene)
        torch.save(pred1,f"{outdir}/pred_DGGAT_{str(i)}.pkl")
        outfile = open(f"{outdir}/edge_weight_{str(i)}.txt", 'w')
        for i in range(edge_index.shape[0]):
            s = node_names[edge_index[i][0]]
            t = node_names[edge_index[i][1]]
            weight = attention_weights[i].item()
            outfile.write('\t'.join([s, t, str(weight)]) + '\n')
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DataDir","-D", type=str, default='./data',help="Input Data Dir")
    parser.add_argument("--dataset","-d", type=str, default='STRINGdb',help="CPDB, STRINGdb, IREF_2015, MULTINET, PCNET")
    parser.add_argument("--Device", type=str,default='cuda')
    parser.add_argument("--MaxEpoch", type=int,default=1000)
    parser.add_argument("--Mode", "-M",type=str,choices=['cross_val','train','predict'])
    parser.add_argument("--DropMethod", "-DM",type=str,choices=['gate','random','nodrop'],default='gate')
    parser.add_argument("--ModelPath", type=str)
    parser.add_argument("--OutDir",'-O',type=str,default='./Out')


    args = parser.parse_args()

    data_dir = args.DataDir
    device = args.Device
    mode = args.Mode
    Epoch = args.MaxEpoch
    outdir = args.OutDir
    nk = args.dataset

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    

    data = torch.load(f"{data_dir}/{nk}.pkl").to(device)
    data.x = torch.from_numpy(data.x).float()
    data.edge_index = torch.from_numpy(data.edge_index)
    data.y = torch.from_numpy(data.y)
    data = data.to(device)
    Y = data.y.float().view(-1,1)
    mask_all = data.mask
    
    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)

    with open(f"{data_dir}/split_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)
        k_sets = k_sets[nk]

    # use VGAE to learn stucture features
    adj = torch.zeros(data.x.shape[0],data.x.shape[0])
    adj[data.edge_index[0],data.edge_index[1]] = 1
    adj = adj.numpy()
    Graph = nx.from_numpy_array(adj)
    node_to_neighbor = {}
    for i in range(adj.shape[0]):
        node_to_neighbor[i] = list(Graph.neighbors(i))
    x_adj = torch.zeros(data.x.shape[0],data.x.shape[0]).float()
    for i in node_to_neighbor.keys():
        x_adj[i,node_to_neighbor[i]] = 1.0
    x_adj = x_adj.to(device)

    data.x_adj = x_adj
    model_z =VGGNN(data.x.shape[0],128, 16).to(device)
    optimizer = torch.optim.Adam(model_z.parameters(), lr=0.001)
    for e in range(1,1000+1):
        optimizer.zero_grad()
        z, loss = model_z(data.x_adj, data.edge_index)
        loss.backward()
        optimizer.step()
        if e%50 ==  0:
            print('epoch: {:03d}, loss: {:.4f}'.format(e, loss))
    model_z.eval()
    z,_ = model_z.forward(data.x_adj, data.edge_index)
    z = z.detach()
    
    data.x = torch.cat([data.x,z],dim=1)

    if mode == 'cross_val':
        cross_val(Epoch,data,Y,k_sets,args.DropMethod,outdir)
    elif mode == 'train':
        train(Epoch,data,Y,mask_all,args.DropMethod,outdir)
    elif mode == 'predict':
        model_path = args.ModelPath
        predict(data,Y,mask_all,args.DropMethod,outdir,model_path,data_dir)

