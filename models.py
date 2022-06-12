import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from torch.nn.parameter import Parameter


device = torch.device("cuda:0")

class GCN_H(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """
    def __init__(self, nfeat, nhid, nclass, num_hparams,
                 activation=lambda x: x, withbn=True, withloop=True, mixmode=False):
        super(GCN_H, self).__init__()
        self.mixmode = mixmode
       
        self.hiddenlayers = nn.ModuleList()

        self.ingc = GraphConvolutionBS_Hyper(nfeat, nhid, num_hparams, activation)
        
        baseblockinput = nhid
        outactivation = lambda x: x
        self.outgc = GraphConvolutionBS_Hyper(baseblockinput, nclass, num_hparams, outactivation)

        self.hid_layer1 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer2 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        if mixmode:
            self.ingc = self.ingc.to(device)
            self.hid_layer1 = self.hid_layer1.to(device)
            self.hid_layer2 = self.hid_layer2.to(device)
            self.outgc = self.outgc.to(device)

    def forward(self, fea, adj, hnet_tensor, hparam_tensor, hdict):
        x = self.ingc(fea, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 0)  # keywords indict the index in Hparams
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer1(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 1)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer2(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 2)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.outgc(x, adj, hnet_tensor)
        x = F.log_softmax(x, dim=1)

        return x


    def get_drop_probs(self, hparam_tensor, hdict, keyword):
        if 'dropout' + str(keyword) in hdict:
            drop_idx = hdict['dropout' + str(keyword)].index
            return hparam_tensor[:, drop_idx]
        else:
            print('Can not get the dropout probs!!!!')
            return 0

    def get_fcdrop_probs(self, hparam_tensor, hdict):
        if 'fcdropout0' not in hdict:
            return (0., 0.)
        fcdrop0_idx = hdict['fcdropout0'].index
        fcdrop1_idx = hdict['fcdropout1'].index
        return (hparam_tensor[:,fcdrop0_idx], hparam_tensor[:, fcdrop1_idx])


    def dropout(self, x, probs, training=False):
        """
        Arguments:
            x (Tensor): whose first dimension has size B
            probs (Tensor): size (B,)
        """
        if not training:
            return x
        if isinstance(probs, float):
            return F.dropout(x, probs, training)
        x_size = x.size()
        x = x.view(x.size(0), -1)
        probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
        mask = (1 - probs).bernoulli().div_(1 - probs)
        return (x * mask).view(x_size)
