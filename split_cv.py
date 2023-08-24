from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch, pickle

def split_CV_set(dataDir='./data/',network_list=None,randseed=0):
    if network_list is None:
        network_list = ['CPDB','IREF_2015','IREF','MULTINET','PCNET','STRINGdb']
    split_sets_all = {}
    for network in network_list:
        data = torch.load(f'{dataDir}/{network}.pkl')
        labeled_idx = torch.arange(data.x.shape[0])[data.mask]
        driver_idx = [i.item() for i in labeled_idx if data.y[i]==True]
        nondriver_idx = [i.item() for i in labeled_idx if data.y[i]==False]
        assert len(set(driver_idx+nondriver_idx))==len(labeled_idx)
        X, y = driver_idx + nondriver_idx, np.hstack(([1]*len(driver_idx), [0]*len(nondriver_idx)))
        
        split_sets_ith = []
        for ith in range(10):
            # StratifiedKFold
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=ith)
            split_sets = []
            for train, test in skf.split(X, y):
                # train/test sorts the sample indices in X list.
                # For each split, we should convert the indices in train/test to names
                train = [X[t] for t in train]
                test = [X[t] for t in test]
                tr_mask = np.array([False]*data.x.shape[0])
                tr_mask[train] = True
                te_mask = np.array([False]*data.x.shape[0])
                te_mask[test] = True
                split_sets.append([tr_mask,te_mask])

            split_sets_ith.append(split_sets)
        split_sets_all[network] = split_sets_ith
            
    return split_sets_all


k_sets = split_CV_set()
with open("./data/split_sets.pkl", 'wb') as handle:
    pickle.dump(k_sets,handle)
