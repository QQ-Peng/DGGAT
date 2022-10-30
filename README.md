# DGGAT
## Dependencies
numpy <br>
pandas <br>
scikit-learn <br> 
scipy <br>
torch==1.12.0+cu102 <br>
torch-geometric==2.1.0 <br>

## Run DGGAT
python main.py -M cross_val -DM gate -O ./Out <br>
python main.py -M train -DM gate -O ./Out <br>
python main.py -M predict -DM gate -O ./Out --ModelPath ./model/model.bin

## Implementation of gating GAT
Our implementation of gating GAT is based on https://github.com/gordicaleksa/pytorch-GAT
