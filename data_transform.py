import h5py, os, pickle
import numpy as np
from torch_geometric.data import Data
import torch

def h5py_to_pyg(dataDir='./EMOGIData/',outDir='./data/'):
    # 将EMOGI处理的数据格式转换为pyg的数据格式
    for file in os.listdir(dataDir):
        if file.endswith('.h5'):
            f = h5py.File(f'{dataDir}/{file}', 'r')
            # print(f'*'*40)
            feature_names = f['feature_names'].asstr()[()][:48]
            gene_names = f['gene_names'].asstr()[()]
            features = f['features'].astype(float)[()][:,:48]
            mask = np.logical_or(np.logical_or(f['mask_train'].astype(bool)[()],f['mask_val'].astype(bool)[()]),f['mask_test'].astype(bool)[()])
            y = np.logical_or(np.logical_or(f['y_train'].astype(bool)[()],f['y_val'].astype(bool)[()]),f['y_test'].astype(bool)[()])
            network = f['network'].astype(int)[()]
            row, col = np.nonzero(network)
            edge_index = np.concatenate([[row],[col]],axis=0)
            data = Data(x=features, y=y, edge_index=edge_index, node_names=gene_names, mask = mask)
            torch.save(data,f"{outDir}/{file.split('.h5')[0]}.pkl")

h5py_to_pyg()