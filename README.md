# HyperU-GCN
This is a PyTorch implementation of the HyperU-GCN model in our paper. 

## Installation
1. Clone the repo:
```
git clone https://github.com/xyang2316/HyperU-GCN
cd HyperU-GCN
```
2. Install the following dependencies, including:
- Python3
- PyTorch
- numpy
- scipy
- sklearn
- networkx

## Datasets
- Cora, Citeseer, Pubmed, 
- Coauthor Physics, Amazon Computer, and Amazon Photo

## Run the model:
Run HyperU_GCN_train.py for different datasets:
- Cora
```
python HyperU_GCN_train.py --dataset=cora --start_dropedge=0.3 --start_dropout=0.1 --start_weightdecay=3e-4 --total_epochs=400 --train_lr=5e-4
```
- Citeseer
```
python HyperU_GCN_train.py --dataset=citeseer --start_dropedge=0.3 --start_dropout=0.08 --start_weightdecay=3e-4 --total_epochs=400 --train_lr=5e-4
```
- Pubmed
```
python HyperU_GCN_train.py --dataset=pubmed --start_dropedge=0.8 --start_dropout=0.05 --start_weightdecay=5e-4 --total_epochs=400 --train_lr=5e-3
```
- Physics
```
python HyperU_GCN_train.py --dataset=ms_academic_phy --start_dropedge=0.3 --start_dropout=0.1 --start_weightdecay=5e-5 --total_epochs=600 --train_lr=5e-4
```
- Computers
```
python HyperU_GCN_train.py --dataset=amazon_electronics_computers --start_dropedge=0.3 --start_dropout=0.05 --start_weightdecay=5e-4 --total_epochs=600 --train_lr=5e-4
```
- Photo
```
python HyperU_GCN_train.py --dataset=amazon_electronics_photo --start_dropedge=0.3 --start_dropout=0.1 --start_weightdecay=5e-5 --total_epochs=600 --train_lr=5e-4
```
Other parameters stays as the default values.


