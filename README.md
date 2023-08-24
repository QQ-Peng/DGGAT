# DGGAT
## Dependencies
numpy <br>
pandas <br>
scikit-learn <br> 
scipy <br>
torch==1.12.0+cu102 <br>
torch-geometric==2.1.0 <br>

## Prepare Data
### Download data from EMOGI
https://github.com/schulter/EMOGI

### Transform data
Put all h5py network data into a folder. <br>
Run data_transform.py for transforming the data to PyG Dataset container.

### Split the cross-validation set
Run split_cv.py


## Run DGGAT
python main.py -M cross_val --dataset {network name} -DM gate -O ./Out <br>
python main.py -M train --dataset {network name} -DM gate -O ./Out <br>
python main.py -M predict --dataset {network name} -DM gate -O ./Out --ModelPath ./model/model_gate.bin

## Implementation of gating GAT
Our implementation of gating GAT is based on https://github.com/gordicaleksa/pytorch-GAT
