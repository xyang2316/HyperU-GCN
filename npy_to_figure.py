from __future__ import division
from __future__ import print_function
from cmath import log

import time
import argparse
import numpy as np
import os
from numpy.core.fromnumeric import _std_dispatcher, size
import torch
import torch.nn.functional as F
import torch.optim as optim
from metric import accuracy, roc_auc_compute_fn
from hyperparameter import *
from utils import load_citation, load_reddit_data
from models import *
from sample import Sampler
from encoder import * #MLP, MLP2
from draw_figures import *
from uncertainty_utlis import *
from datetime import datetime
import uuid
import copy

debug_save_pth = "./save/train_double_ms_academic_phy.npy"
saved_info =  np.load(debug_save_pth, allow_pickle=True).item()
print(saved_info.keys())

loss_train = saved_info["train_loss"]
loss_val = saved_info["val_loss"]
acc_train = saved_info["train_acc"]
acc_val = saved_info["val_acc"]

print(np.array(loss_train).shape)
print(np.array(loss_val).shape)
print(np.array(acc_train).shape)
print(np.array(acc_val).shape)

double_figure(loss_train, acc_train)
# double_figure(loss_val, acc_val)

debug_save_pth = "./save/train_entropy_ms_academic_phy.npy"
saved_info =  np.load(debug_save_pth, allow_pickle=True).item()
print(saved_info.keys())

h_uncertainty = saved_info["h_u"]
d_uncertainty = saved_info["data_u"]
t_uncertainty = saved_info["total_u"]

print(np.array(h_uncertainty).shape)
print(np.array(d_uncertainty).shape)
print(np.array(t_uncertainty).shape)

plt.figure(figsize=(6, 6))
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["font.weight"] = "bold"
plt.plot([i+1 for i in range(len(h_uncertainty) - 30)], h_uncertainty[30:], linewidth=2, color="maroon", label="Hyper uncertainty")
plt.plot([i+1 for i in range(len(d_uncertainty) - 30)], d_uncertainty[30:], linewidth=2, color="green", label="Data uncertatiny")
plt.plot([i+1 for i in range(len(t_uncertainty) - 30)], t_uncertainty[30:], linewidth=2, color="royalblue", label="Total uncertainty")
# plt.plot([i+1 for i in range(len(h_uncertainty))], h_uncertainty, linewidth=1.5, color="maroon", label="Hyper uncertainty")
# plt.plot([i+1 for i in range(len(d_uncertainty))], d_uncertainty, linewidth=1.5, color="green", label="Data uncertatiny")
# plt.plot([i+1 for i in range(len(t_uncertainty))], t_uncertainty, linewidth=1.5, color="royalblue", label="Total uncertainty")

plt.tick_params(labelsize=14)
plt.legend(fontsize=17)
plt.grid(alpha=0.6)
plt.xticks(np.arange(0, len(h_uncertainty), 50))
plt.yticks(np.arange(0, 0.3, 0.05))
plt.xlabel("Training Epochs", fontsize=15)
plt.ylabel("Entropy", fontsize=15)
plt.savefig("Final_Hyper_U.png", bbox_inches='tight', format='png', dpi=300, pad_inches=0)



