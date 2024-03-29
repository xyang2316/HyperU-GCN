# HyperU-GCN 
This is a PyTorch implementation of the HyperU-GCN model in our following paper:

Xueying Yang, Jiamian Wang, [Xujiang Zhao](https://zxj32.github.io/), [Sheng Li](http://sheng-li.org/) and [Zhiqiang Tao](https://ztao.cc/), "[Calibrate Automated Graph Neural Network via Hyperparameter Uncertainty](https://dl.acm.org/doi/pdf/10.1145/3511808.3557556)", CIKM 2022.

## Introduction
<p align="justify">
Automated graph learning has drawn widespread research attention due to its great potential to reduce human efforts when dealing with graph data, among which hyperparameter optimization (HPO) is one of the mainstream directions with promising progress. However, how to obtain reliable and trustworthy prediction results with automated graph neural networks (GNN) is still quite underexplored. To this end, we investigate automated GNN calibration by marrying uncertainty estimation to the HPO problem. We propose a hyperparameter uncertainty-induced graph convolutional network (HyperU-GCN) with a bilevel formulation, where the upper-level problem explicitly reasons uncertainties by developing a probabilistic hypernetworks through a variational Bayesian lens, while the lower-level problem learns how the GCN weights respond to a hyperparameter distribution. By squeezing model uncertainty into the hyperparameter space, the proposed HyperU-GCN could achieve calibrated predictions in a similar way to Bayesian model averaging over hyperparameters. Extensive experimental results on six public datasets were provided in terms of node classification accuracy and expected calibration error (ECE), demonstrating the effectiveness of our approach compared with several state-of-the-art uncertainty-aware and calibrated GCN methods.
</p>

<p align="center">
 <img width="800" alt="image" src="https://user-images.githubusercontent.com/55004948/195007873-3fc18e33-7426-4594-a7d1-110b6b0d4d5c.png">
</p>

## Installation
1. Clone the repo:
```
git clone https://github.com/xyang2316/HyperU-GCN
cd HyperU-GCN
```
2. Install the following dependencies, including:
- Python 3.9.7
- PyTorch 1.11
- numpy 1.21.2
- scipy 1.7.3
- sklearn 1.0.2
- networkx 2.6.3

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

## Results
The expeiment results in our paper:
<p align="center">
 <img width="945" alt="image" src="https://user-images.githubusercontent.com/55004948/195495959-b05d82ba-aaa6-40eb-8c21-0b0d8bab0425.png">
</p>

## Citation
If you find the code helpful in your resarch or work, please cite our paper: 
```
@inproceedings{yang2022calibrate,
  title={Calibrate Automated Graph Neural Network via Hyperparameter Uncertainty},
  author={Yang, Xueying and Wang, Jiamian and Zhao, Xujiang and Li, Sheng and Tao, Zhiqiang},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4640--4644},
  year={2022}
}
```
