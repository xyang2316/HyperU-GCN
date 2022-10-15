from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import uuid
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from hyperparameter import *
from models import *
from sample import Sampler
from draw.draw_figures import *
from utils.util import *

# Training settings
parser = argparse.ArgumentParser()
 
# Tuning options
parser.add_argument('--tune_scales', '-tscl', action='store_true', default=True, help='whether to tune scales of perturbations by penalizing entropy on training set')
parser.add_argument('--tune_dropedge', '-tdrope', action='store_true', default=True, help='whether to tune edge dropout rates')
parser.add_argument('--tune_dropout', '-tdrop', action='store_true', default=True, help='whether to tune dropout rates (per layer)')
parser.add_argument('--tune_weightdecay', '-tweidec', action='store_true', default=True, help='whether to tune weight decay')
# Initial hyperparameter settings
parser.add_argument('--start_dropedge', '-drope', type=float, default=0.3, help='starting edge dropout rate')
parser.add_argument('--start_dropout', '-drop', type=float, default=0.1, help='starting dropout rate') 
parser.add_argument('--start_weightdecay', '-weidec', type=float, default=5e-5, help='starting weightdecay rate') 
# Optimization hyperparameters
parser.add_argument('--total_epochs', '-totep', type=int, default=400, help='number of training epochs to run for (warmup epochs are included in the count)')
parser.add_argument('--warmup_epochs', '-wupep', type=int, default=30, help='number of warmup epochs to run for before tuning hyperparameters')
parser.add_argument('--train_lr', '-tlr', type=float, default=5e-4, help='learning rate on parameters') 
parser.add_argument('--valid_lr', '-vlr', type=float, default=3e-3, help='learning rate on hyperparameters') 
parser.add_argument('--encoder_lr', '-elr', type=float, default=1e-4, help='learning rate on hyperparameters') 
parser.add_argument('--scale_lr', '-slr', type=float, default=1e-3, help='learning rate on scales (used if tuning scales)')
parser.add_argument('--momentum', '-mom', type=float, default=0.9, help='amount of momentum on usual parameters')
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
parser.add_argument("--normalization", default="BingGeNormAdj", help="The normalization on the adj matrix.")
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi).")

args = parser.parse_args()

# set gpu device
os.environ["CUDA_VISIVLE_DEVICES"] = '0'
torch.cuda.set_device(0) 

# init logs for plots
train_loss_log = []
train_acc_log = []
test_acc_log = []
test_loss_log = []
test_val_acc_log = []
test_val_loss_log = []

h_uncertainty = []
t_uncertainty = []
d_uncertainty = []

# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

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
labels, idx_train, idx_val, idx_test, idx_test_ood, idx_test_id = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

###############################################################################
# Model/Optimizer
###############################################################################
# specify the number of dropout
num_drops = 3  
htensor, _, hdict = create_hparams(args, num_drops, device)
num_hparams = htensor.size()[0]
best_htensor = htensor
print('num_hparams is ', num_hparams)

# create encoder net model
encoder = MLP(num_hparams, device).cuda()

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

###############################################################################
# Evaluation
###############################################################################
def evaluate_test(test_adj, test_fea, index, is_warmup=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    if not is_warmup:
        checkpoint = torch.load(checkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # htensor = encoder(htensor).to(device) # add this if you want to sample htensor during evaluation 
    hparam_tensor = hparam_transform(htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(htensor.repeat(len(labels), 1), hdict)

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    output_prob = torch.exp(output)
    loss_test = F.nll_loss(output[index], labels[index])
    acc_test = accuracy(output[index], labels[index])
   
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    return loss_test, acc_test, output_prob

def evaluate(test_adj, test_fea, index, is_warmup=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    if not is_warmup:
        checkpoint = torch.load(checkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # htensor = encoder(best_htensor).to(device) # add this if you want to sample htensor during evaluation 
    
    hparam_tensor = hparam_transform(best_htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(best_htensor.repeat(len(labels), 1), hdict)

    # hparam_tensor = hparam_transform(htensor.repeat(len(labels), 1), hdict) # add these to use last(previous) epoch htensor
    # hnet_tensor = hnet_transform(htensor.repeat(len(labels), 1), hdict)

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    output_prob = torch.exp(output)
    loss_test = F.nll_loss(output[index], labels[index])
    acc_test = accuracy(output[index], labels[index])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    return loss_test, acc_test, output_prob

###############################################################################
# Optimization step
###############################################################################
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

    all_htensor = new_htensor.repeat(sampler.ndata, 1)  
    
    hparam_tensor = hparam_transform(all_htensor, hdict)
    hnet_tensor = hnet_transform(all_htensor[:sampler.ndata], hdict)
    
    # Calculate hyper uncertainty
    if not hyper:
        avg_times = 10
        t_outputs = []
        
        # total uncertainty:
        for _ in range(avg_times):
            h_single = encoder(htensor).to(device) 
            t_htensor = h_single.repeat(sampler.ndata, 1).cuda() 
        
            t_hparam_tensor = hparam_transform(t_htensor, hdict)
            t_hnet_tensor = hnet_transform(t_htensor[:sampler.ndata], hdict)
            (train_adj, train_fea) = sampler.randomedge_sampler_hp(t_hparam_tensor, hdict,
                                                            normalization=args.normalization, cuda=args.cuda)
            t_output = model(train_fea, train_adj, t_hnet_tensor, t_hparam_tensor, hdict)
            t_outputs.append(torch.exp(t_output).detach().cpu().numpy())
        t_out_avg = torch.tensor(np.mean(t_outputs, axis=0))
        total_un,  total_un_class= compute_entropy(t_out_avg, index, nclass)
        t_uncertainty.append(total_un)
        
        # data uncertainty:
        data_un_s = []
        data_un_classes = []
        for _ in range(avg_times):
            h_single = encoder(htensor).to(device) 
            d_all_htensor = h_single.repeat(sampler.ndata, 1).cuda()  
        
            d_hparam_tensor = hparam_transform(d_all_htensor, hdict)
            d_hnet_tensor = hnet_transform(d_all_htensor[:sampler.ndata], hdict)
            (train_adj, train_fea) = sampler.randomedge_sampler_hp(d_hparam_tensor, hdict,
                                                                normalization=args.normalization, cuda=args.cuda)
            d_output = model(train_fea, train_adj, d_hnet_tensor, d_hparam_tensor, hdict)
            data_un,  data_un_class = compute_entropy(torch.exp(d_output), index, nclass)
            data_un_s.append(data_un)
            data_un_classes.append(data_un_class)

        data_uncertainty = np.mean(data_un_s, axis=0)
        data_uncertainty_class = np.mean(data_un_classes, axis=0)
        eps_class = total_un_class - data_uncertainty_class
        eps_un = np.sum(eps_class, axis=0) 
        d_uncertainty.append(data_uncertainty)
        h_uncertainty.append(eps_un)
    
    (train_adj, train_fea) = sampler.randomedge_sampler_hp(hparam_tensor, hdict,
                                                           normalization=args.normalization, cuda=args.cuda)
    output = model(train_fea, train_adj, hnet_tensor, hparam_tensor, hdict)
    loss_train = F.nll_loss(output[index], labels[index])
    acc_train = accuracy(output[index], labels[index])
    
    if not hyper:
        loss_train.backward()
        gcn_optimizer.step()
        weight_decay = get_weightdecay_pro(hparam_tensor, hdict)
        gcn_optimizer.param_groups[0]['weight_decay'] = weight_decay.item()
    else:
        loss_validation = loss_train - F.kl_div(new_htensor.log_softmax(0), torch.empty(htensor.size()).normal_(mean=0, std=1).to(device))
        loss_validation.backward()
        encoder_optimizer.step()
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
# Warm-up Loop
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
   
    wup_step += 1
    global_step += 1
    train_epoch += 1

    (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    val_loss, val_acc, _ = evaluate(val_adj, val_fea, idx_val)

scheduler = optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

###############################################################################
# Alternatively training and validation
###############################################################################
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
            test_loss, test_acc, _ = evaluate_test(test_adj, test_fea, idx_test)
            test_val_loss, test_val_acc, _ = evaluate_test(test_adj, test_fea, idx_val)
            with torch.no_grad():
                htensor.copy_(new_htensor)
           
            print('Tra_Epoch: {}  |  train_loss {:.4f}  |  train_acc {:.4f}'.format(global_step, train_loss, train_acc))
            
            train_step += 1
            
            train_loss_log.append(train_loss.item())
            train_acc_log.append(train_acc.item())
            test_acc_log.append(test_acc.item()) 
            test_loss_log.append(test_loss.item()) 
            test_val_acc_log.append(test_val_acc.item())
            test_val_loss_log.append(test_val_loss.item())
            
        # do a step on the validation set.
        else:
            model.eval()
            
            val_loss, val_acc, new_htensor = optimization_step(htensor, sampler, idx_val, hyper)
            test_loss, test_acc, _ = evaluate_test(test_adj, test_fea, idx_test)
            with torch.no_grad():
                htensor.copy_(new_htensor)
            valid_step += 1
            print('Val_Epoch: {}  |  val_loss {:.4f}  |  val_acc {:.4f}'.format(global_step, val_loss, val_acc))
            
            
        if val_acc > worst:
            worst = val_acc
            best_epoch = global_step
            best_acc = val_acc
            best_htensor = copy.deepcopy(htensor)
            torch.save({'model_state_dict': model.state_dict(),
                        'encoder_state_dict': encoder.state_dict()},checkpt_file)
    
        global_step += 1    

except KeyboardInterrupt:
    print('=' * 80)
    print('Exiting from training early')

###############################################################################
# Testing
###############################################################################
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
val_loss, val_acc, _ = evaluate(test_adj, test_fea, idx_val)

# last-epoch results
test_loss, test_acc, test_output_logit1 = evaluate_test(test_adj, test_fea, idx_test)
print('=' * 89)
print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | test_acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc))
print('=' * 89)

# enable these for best model according to validation accuarcy
# test_loss, test_acc, test_output_logit2 = evaluate(test_adj, test_fea, idx_test, False)
# print("Best epoch: ", best_epoch)
# print('=' * 89)
# print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | test_acc {:8.5f}'.format(
#          val_loss, val_acc, test_loss, test_acc))
# print('=' * 89)

###############################################################################
# Saving model performance
###############################################################################
test_dict = {"test_loss": test_loss_log, "test_acc": test_acc_log, "val_loss": test_val_loss_log, "val_acc": test_val_acc_log, "train_loss": train_loss_log, "train_acc":train_acc_log}
np.save("./save/test_"+args.dataset, test_dict)
entrop_dict = {"total_u": t_uncertainty, "data_u": d_uncertainty, "h_u": h_uncertainty}
np.save("./save/entropy_"+args.dataset, entrop_dict)

###############################################################################
# Reiability diagram: load last-epoch model with best_htensor/last htensor
###############################################################################
labels_oneh = convert2one_hot(labels, nclass, device)
labels_oneh = labels_oneh[idx_test].cpu().numpy()
preds = test_output_logit1[idx_test]
preds = preds.cpu().detach().numpy()
ECE = draw_reliability_graph(labels_oneh, preds, args.dataset, "1_HyperU-GCN", args.task_type)
print("Final ECE:", ECE)

###############################################################################
# Reiability diagram: load best-validaition model with best_htensor
###############################################################################
# labels_oneh = convert2one_hot(labels, nclass, device)
# labels_oneh = labels_oneh[idx_test].cpu().numpy()
# preds = test_output_logit2[idx_test]
# preds = preds.cpu().detach().numpy()
# ECE = draw_reliability_graph(labels_oneh, preds, args.dataset, "HyperU-GCN", args.task_type)
# print("Final ECE:", ECE)