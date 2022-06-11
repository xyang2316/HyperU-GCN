from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
from numpy.core.fromnumeric import _std_dispatcher, size
import torch
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter

from metric import accuracy, roc_auc_compute_fn
from hyperparameter import *
from utils import load_citation, load_reddit_data
from models import *
from sample import Sampler
from encoder import MLP, MLP2
from draw_figures import *
from uncertainty_utlis import *
from datetime import datetime
import uuid
import copy
# from logger import Logger
# "ms_academic_phy", 'amazon_electronics_computers', 'amazon_electronics_photo',  ("ms_academic_cs")

# Training settings
parser = argparse.ArgumentParser()
 
# Tuning options
parser.add_argument('--tune_scales', '-tscl', action='store_true', default=True, help='whether to tune scales of perturbations by penalizing entropy on training set')
parser.add_argument('--tune_dropedge', '-tdrope', action='store_true', default=True, help='whether to tune edge dropout rates')
parser.add_argument('--tune_dropout', '-tdrop', action='store_true', default=True, help='whether to tune dropout rates (per layer)')
parser.add_argument('--tune_weightdecay', '-tweidec', action='store_true', default=True, help='whether to tune weight decay')
# Initial hyperparameter settings
parser.add_argument('--start_dropedge', '-drope', type=float, default=0.3, help='starting edge dropout rate')#0.2 0.3 0.8
parser.add_argument('--start_dropout', '-drop', type=float, default=0.1, help='starting dropout rate') # 0.1 0.05 0.1
parser.add_argument('--start_weightdecay', '-weidec', type=float, default=5e-5, help='starting weightdecay rate') #0.0001, 5e-4 0.0001
# Optimization hyperparameters
parser.add_argument('--total_epochs', '-totep', type=int, default=600, help='number of training epochs to run for (warmup epochs are included in the count)')
parser.add_argument('--warmup_epochs', '-wupep', type=int, default=30, help='number of warmup epochs to run for before tuning hyperparameters')
parser.add_argument('--train_lr', '-tlr', type=float, default=5e-4, help='learning rate on parameters') #try 5e-3 for pubmed
parser.add_argument('--valid_lr', '-vlr', type=float, default=3e-3, help='learning rate on hyperparameters') #3e-3
parser.add_argument('--encoder_lr', '-elr', type=float, default=1e-4, help='learning rate on hyperparameters') 
parser.add_argument('--scale_lr', '-slr', type=float, default=1e-3, help='learning rate on scales (used if tuning scales)')
parser.add_argument('--momentum', '-mom', type=float, default=0.9, help='amount of momentum on usual parameters') #).9
parser.add_argument('--train_steps', '-tstep', type=int, default=2, help='number of batches to optimize parameters on training set')
parser.add_argument('--valid_steps', '-vstep', type=int, default=1, help='number of batches to optimize hyperparameters on validation set')

# Regularization hyperparameters
parser.add_argument('--entropy_weight', '-ewt', type=float, default=1e-5, help='penalty applied to entropy of perturbation distribution')
parser.add_argument('--perturb_scale', '-pscl', type=float, default=0.5, help='scale of perturbation applied to continuous hyperparameters')

# Training parameter 
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--mixmode", action="store_true", default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument('--dataset', default="citeseer", help="The data set")
parser.add_argument('--datapath', default="./data/", help="The data path.")
parser.add_argument("--early_stopping", type=int, default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument('--OOD_detection', type=int, default=0, help="0 for Misclassification, 1 for OOD detection.")

# Model parameter
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--withbn', action='store_true', default=False, help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False, help="Enable loop layer GCN")
parser.add_argument("--normalization", default="BingGeNormAdj", help="The normalization on the adj matrix.")
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

args = parser.parse_args()

os.environ["CUDA_VISIVLE_DEVICES"] = '0'
torch.cuda.set_device(0) #
train_loss_log = []
train_acc_log = []
val_loss_log = []
val_acc_log = []
htensor_log = []
mean_log = []
std_log = []
epsilon_log = []
P = []
test_acc_log = []
test_loss_log = []

# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

#new
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt' 
print(checkpt_file)

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Data Loading/Processing
###############################################################################
sampler = Sampler(args.dataset, args.datapath, args.task_type, args.OOD_detection)
# get labels and indexes
labels, idx_train, idx_val, idx_test, idx_test_ood, idx_test_id = sampler.get_label_and_idxes(args.cuda) # add test OOD ID
print(len(idx_train), len(idx_val), len(idx_test), len(idx_test_ood), len(idx_test_id)) #80 333 1000 316

# time.sleep(50)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

###############################################################################
# Model/Optimizer
###############################################################################
num_drops = 3  # this parameter decide the number of dropout
htensor, _, hdict = create_hparams(args, num_drops, device)
#print(htensor.size(), hscale.size(), len(hdict)) #torch.Size([5]) torch.Size([5]) 5
num_hparams = htensor.size()[0]
best_htensor = htensor
print('num_hparams is ', num_hparams)

# create encoder net model
encoder = MLP(num_hparams, device).cuda()
print(list(encoder.parameters()))

# create hyper GCN model
model = GCN_H(nfeat=nfeat, nhid=args.hidden, nclass=nclass, num_hparams=num_hparams,
              activation=F.relu)
model = model.cuda()
total_params = sum(param.numel() for param in model.parameters())
print('Args: ', args)
print('total_params: ', total_params)

gcn_optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.start_weightdecay)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
scheduler = optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# encoder_scheduler = optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

labels_oneh = convert2one_hot(labels, nclass, device)
###############################################################################
# Evaluation
###############################################################################
def evaluate(test_adj, test_fea, index, is_warmup=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    if not is_warmup:
        checkpoint = torch.load(checkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    new_htensor = encoder(best_htensor).to(device) #
    
    hparam_tensor = hparam_transform(best_htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(best_htensor.repeat(len(labels), 1), hdict)

    # hparam_tensor = hparam_transform(htensor.repeat(len(labels), 1), hdict)
    # hnet_tensor = hnet_transform(htensor.repeat(len(labels), 1), hdict)

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    output_prob = torch.exp(output)
    loss_test = F.nll_loss(output[index], labels[index])
    acc_test = accuracy(output[index], labels[index])

    print("A===" * 20)
    print("evalutation htensor:", hnet_tensor[0])
    print("evalutation hparam:", hparam_tensor[0]) #torch.Size([3327, 5])
    for name, param in model.named_parameters():
        print("model: ", name, param)
        break
    print("B===" * 20)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    return loss_test, acc_test, output_prob

def evaluate_ood(test_adj, test_fea, index, id_index, is_warmup=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    if not is_warmup:
        model.load_state_dict(torch.load(checkpt_file))
    model.eval()

    hparam_tensor = hparam_transform(best_htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(best_htensor.repeat(len(labels), 1), hdict)

    print(htensor)
    print(hparam_tensor[0])

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    output_prob = torch.exp(output)
    loss_test = F.nll_loss(output[id_index], labels[id_index])
    acc_test = accuracy(output[id_index], labels[id_index])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          # "auc= {:.4f}".format(auc_test),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    print("return evaluate")
    return loss_test, acc_test, output_prob
###############################################################################
# Optimization step
###############################################################################

def EDL_loss(output, Y, device, hyper):
    logits = output
    print(logits.size())
    evidence = torch.exp(torch.clamp(logits, -10, 10))
    alpha = torch.add(evidence, 1)
    S = torch.sum(alpha)
    E = alpha - 1
    m = alpha / S #p_k hat
    p = convert2one_hot(Y, nclass, device)

    A = torch.sum((p-m)**2) 
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1))) 
    
    temp = global_step / (10 * args.train_steps) if not hyper else global_step / (10 * args.valid_steps)
    annealing_coef = min(1.0, temp)
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp, device)
    return (A + B) + C

def KL(alpha, device):
    K = nclass
    beta = torch.ones(K, dtype = torch.float32).to(device)
    S_alpha = torch.sum(alpha).to(device)
    S_beta = torch.sum(beta).to(device)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha)) #Computes the log of the absolute value of Gamma(x) element-wise.
    lnB_uni = torch.sum(torch.lgamma(beta)) - torch.lgamma(S_beta) 
    
    dg0 = torch.digamma(S_alpha).to(device)
    dg1 = torch.digamma(alpha).to(device) #Computes Psi, the derivative of Lgamma (the log of the absolute value of Gamma(x)), element-wise.
    kl = torch.sum((alpha - beta)*(dg1-dg0)).to(device) + lnB.to(device) + lnB_uni.to(device)
    kl.to(device)
    return kl


def optimization_step(htensor, sampler, index, hyper=False):
    if not hyper:
        model.train()
    else:
        model.eval()
    gcn_optimizer.zero_grad()
    encoder_optimizer.zero_grad()

    if hyper:
        new_htensor = encoder(htensor).to(device)
    else:
        new_htensor = htensor

    all_htensor = new_htensor.repeat(sampler.ndata, 1)  #new_htensor
    
    hparam_tensor = hparam_transform(all_htensor, hdict)
    hnet_tensor = hnet_transform(all_htensor[:sampler.ndata], hdict)

    (train_adj, train_fea) = sampler.randomedge_sampler_hp(hparam_tensor, hdict,
                                                           normalization=args.normalization, cuda=args.cuda)
    output = model(train_fea, train_adj, hnet_tensor, hparam_tensor, hdict)
    loss_train = F.nll_loss(output[index], labels[index])
    acc_train = accuracy(output[index], labels[index])

    a_encoder_param_list = list(encoder.parameters())
    a = a_encoder_param_list[1].clone()
 
    if not hyper:
        loss_train.backward()
        gcn_optimizer.step()
        weight_decay = get_weightdecay_pro(hparam_tensor, hdict)
        gcn_optimizer.param_groups[0]['weight_decay'] = weight_decay.item()
    else:
        loss_validation = loss_train - F.kl_div(new_htensor.log_softmax(0), torch.empty(htensor.size()).normal_(mean=0, std=1).to(device))
        loss_validation.backward()

        encoder_optimizer.step()

        b_encoder_param_list = list(encoder.parameters())
        b = b_encoder_param_list[1].clone()
        print("a==b???",torch.equal(a.data, b.data))
    return loss_train, acc_train, new_htensor


def get_weightdecay_pro(hparam_tensor, hdict):
    if 'weightdecay' in hdict:
        drop_idx = hdict['weightdecay'].index
    else:
        print('No dropedge !!!!')
        return 0
    return hparam_tensor[0, drop_idx]


def get_dropedge_pro(hparam_tensor, hdict):
    if 'dropedge' in hdict:
        drop_idx = hdict['dropedge'].index
    else:
        print('No dropedge !!!!')
        return 0
    return hparam_tensor[:, drop_idx]
###############################################################################
# Training Loop
###############################################################################
train_step = valid_step = global_step = wup_step = 0
train_epoch = valid_epoch = 0
test_step = 0

#randomedge sampling
(train_adj, train_fea) = sampler.randomedge_sampler(percent=args.start_dropedge, normalization=args.normalization,
                                                    cuda=args.cuda)
print("WARNUP")
model.train()
while train_epoch < args.warmup_epochs:
    
    _, _, new_htensor = optimization_step(htensor, sampler, idx_train)
    with torch.no_grad():
        htensor.copy_(new_htensor)
   
    htensor_log.append(htensor.tolist()) #
    wup_step += 1
    global_step += 1
    train_epoch += 1

    (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    val_loss, val_acc, _ = evaluate(val_adj, val_fea, idx_val)
    val_loss_log.append(val_loss.item())#
    val_acc_log.append(val_acc.item())#

scheduler = optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
print("MAIN")
best = 999999999
worst = -999999
best_epoch = 0
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
try:
    # Enter main training loop. Alternate between optimizing on training set for
    # args.train_steps and on validation set for args.valid_steps.
    while global_step < args.total_epochs:
        # check whether we should use training or validation set
        cycle_pos = (train_step + valid_step) % (args.train_steps + args.valid_steps)
        hyper = cycle_pos >= args.train_steps

        # do a step on the training set.
        if not hyper:
            model.train()
            
            train_loss, train_acc, new_htensor = optimization_step(htensor, sampler, idx_train, hyper)
            with torch.no_grad():
                htensor.copy_(new_htensor)
           
            # train_loss, train_acc = optimization_step(train_epoch, train_adj, train_fea, idx_train)
            print('Tra_Epoch: {}  |  train_loss {:.4f}  |  train_acc {:.4f}'.format(global_step, train_loss, train_acc))
            train_loss_log.append(train_loss.item())#
            train_acc_log.append(train_acc.item())#
            train_step += 1
        # do a step on the validation set.
        else:
            model.eval()
            
            val_loss, val_acc, new_htensor = optimization_step(htensor, sampler, idx_val, hyper)
            with torch.no_grad():
                htensor.copy_(new_htensor)
    
            print('Val_Epoch: {}  |  val_loss {:.4f}  |  val_acc {:.4f}'.format(global_step, val_loss, val_acc))
            val_loss_log.append(val_loss.item())#
            val_acc_log.append(val_acc.item())#
            valid_step += 1
            P.append((val_acc, val_loss, global_step))
            
            
        # if global_step == args.total_epochs and val_loss < best:
        # if val_loss < best:
        #     best_epoch = global_step
        #     best_acc = val_acc
        #     best_htensor = htensor
        #     torch.save(model.state_dict(), checkpt_file)
        if val_acc > worst:
            worst = val_acc
            best_epoch = global_step
            best_acc = val_acc
            # best_htensor = htensor
            best_htensor = copy.deepcopy(htensor)
            print("@@@validattion best_htensor: ", global_step, htensor)
            for name, param in model.named_parameters():
                print("model: ", name, param)
                break
            torch.save({'model_state_dict': model.state_dict(),
                        'encoder_state_dict': encoder.state_dict()},checkpt_file)
    
        global_step += 1
        htensor_log.append(htensor.tolist())
        

except KeyboardInterrupt:
    print('=' * 80)
    print('Exiting from training early')


(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

# Run on val and test data.
# val_loss, val_acc, _ = evaluate(test_adj, test_fea, idx_val, False)
# val_loss, val_acc, _ = evaluate(test_adj, test_fea, idx_val, True)

# #######
val_loss, val_acc, _ = evaluate(test_adj, test_fea, idx_val)
test_loss, test_acc, test_output_logit1 = evaluate(test_adj, test_fea, idx_test)

print('=' * 89)
print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | test_acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc))
print('=' * 89)
# time.sleep(10000)
#######

Baye_res = []
test_acc_sum = 0
if args.OOD_detection == 0:
    for i in range(1): #
        test_idx_i = np.random.choice(idx_test.cpu(), size=1000, replace=False)
        test_idx_filtered = [ele for ele in test_idx_i if ele in idx_test_id]
        # test_loss, test_acc, test_output_logit = evaluate(test_adj, test_fea, test_idx_filtered, True) #cora
        test_loss, test_acc, test_output_logit2 = evaluate(test_adj, test_fea, test_idx_filtered, False)
        test_acc_sum += test_acc
elif args.OOD_detection == 1:
    for i in range(10):
        test_idx_i = np.random.choice(idx_test_id.cpu(), size=1000, replace=False)
        test_loss, test_acc, test_output_logit2 = evaluate_ood(test_adj, test_fea, idx_test, test_idx_i, False)
        test_acc_sum += test_acc
Baye_res.append(test_output_logit2.detach().cpu().numpy())
test_acc_avg = test_acc_sum / 1.0

print("Best epoch: ", best_epoch)
print('=' * 89)
print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | average_test_acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc_avg))
print('=' * 89)



test_dict = {"test_loss": test_loss_log, "test_acc": test_acc_log}
np.save("./save/test_"+args.dataset, test_dict)
###############################################################################
# + Uncertainty
###############################################################################

if args.OOD_detection == 1:
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        roc, pr = OOD_Detection_citation(Baye_res, args.dataset, "STN-GCN")
    elif args.dataset in ["amazon_electronics_computers", "amazon_electronics_photo", "ms_academic_cs", "ms_academic_phy"]:
        roc, pr = OOD_Detection_npz(Baye_res, args.dataset, "STN-GCN")
    
    print("roc", roc)
    print("pr", pr)
    print("OOD_Detection AUROC: ", "Vacuity = ", roc[0], "Dissonance ", roc[1], "Aleatoric = ", roc[2], "Epistemic ", roc[3], "Entropy = ", roc[4])
    print("OOD_Detection AUPR: ", "Vacuity = ", pr[0], "Dissonance ", pr[1], "Aleatoric = ", pr[2], "Epistemic ", pr[3], "Entropy = ", pr[4])



###############################################################################
# Reiability diagram 1: not load model, best_htensor
###############################################################################
import torchmetrics
preds = test_output_logit1[idx_test]
target = labels[idx_test] 
ece1 = torchmetrics.functional.calibration_error(preds, target, n_bins=10, norm='l1')
print("not sure ECE1:", ece1)



preds = test_output_logit1[idx_test]
labels_oneh = labels_oneh[idx_test].cpu().numpy()
preds = preds.cpu().detach().numpy()
print("preds", len(preds))
print("labels", len(labels_oneh))

ECE = draw_reliability_graph(labels_oneh, preds, args.dataset, "1_MLP+STN-GCN", args.task_type)
print("Final ECE:", ECE)
###############################################################################
# Reiability diagram 2: load model, best_htensor
###############################################################################
preds = test_output_logit2[idx_test]
ece2 = torchmetrics.functional.calibration_error(preds, target, n_bins=10, norm='l1')
print("not sure ECE2:", ece2)

preds = test_output_logit2[idx_test]
# labels_oneh = labels_oneh[idx_test].cpu().numpy()
preds = preds.cpu().detach().numpy()
print("preds", len(preds))
print("labels", len(labels_oneh))

ECE = draw_reliability_graph(labels_oneh, preds, args.dataset, "2_MLP+STN-GCN", args.task_type)
print("Final ECE:", ECE)

###############################################################################
# loss diagram
###############################################################################
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


# fig, axs = plt.subplots(3, 2, figsize=(15, 15))
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))
# axs[0, 0].plot([i + 1 for i in range(len(val_loss_log))], val_loss_log)
# axs[0, 0].set_title("validation loss")
# axs[0, 1].plot([i + 1 for i in range(len(train_loss_log))], train_loss_log)
# axs[0, 1].set_title("training loss")

# axs[1, 0].plot([i + 1 for i in range(len(train_loss_log))], train_loss_log)
# axs[1, 0].plot([i + 1 for i in range(len(val_loss_log))], val_loss_log)
# axs[1, 0].set_title("training and validation loss")

# labels = ["h1", "h2", "h3", "h4","h5"]
# l = axs[1, 1].plot([i + 1 for i in range(len(htensor_log))], htensor_log, label=labels)
# axs[1, 1].legend((line for line in l),(label for label in labels), loc="lower right")
# axs[1, 1].set_title("htensor")

# fig.tight_layout()
# today_date = datetime.today().strftime('%Y-%m-%d')
# plt.savefig("MLP")
# plt.savefig(today_date + "_MLP_STNGCN_"+ args.dataset +"test_acc_")

print("best_htensor: ", best_htensor)
print("final htensor", htensor)
print("best_val_acc: ", best_acc)
print("P(val_acc): ", max(P, key = lambda t: t[0]))
print("P(val_loss): ", min(P, key = lambda t: t[1]))
print("P()last: ", P[-1])
# for name, param in encoder.named_parameters():
#     print(name, param)