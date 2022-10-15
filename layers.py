import math
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn


class GraphConvolutionBS_Hyper(Module):
    """
    hyper version GCN Layer with BN, Self-loop and Res connection.
    1. activation is used for res.
    """
    def __init__(self, in_features, out_features, num_hparams, activation=lambda x: x, bias=True):
        super(GraphConvolutionBS_Hyper, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hparams = num_hparams
        self.sigma = activation
        self.bias = bias

        # Parameter setting.
        self.elem_weight = Parameter(torch.FloatTensor(in_features, out_features)) #W_ele
        self.hnet_weight = Parameter(torch.FloatTensor(out_features, in_features)) #W_lambda

        if self.bias:
            self.elem_bias = Parameter(torch.FloatTensor(out_features)) #b_ele
            self.hnet_bias = Parameter(torch.FloatTensor(out_features)) #b_lambda
        else:
            self.register_parameter('elem_bias', None)
            self.register_parameter('hnet_bias', None)

        self.htensor_to_scalars = nn.Linear(num_hparams, 2 * out_features, bias=False)
        self.elem_scalar = nn.Parameter(torch.ones(1))
        self.init_params()

    def forward(self, input, adj, htensor):
        """
        :param input: size should be (B, D)
        :param adj: size should be (B, B)
        :param htensor: size should be (B, num_hparams)
        """
        support = torch.mm(input, self.elem_weight)
        output = torch.spmm(adj, support)  # correct

        if self.bias is not None:
            output = output + self.bias
        output *= self.elem_scalar

        hnet_scalars = self.htensor_to_scalars(htensor)
        hnet_wscalars = hnet_scalars[:, :self.out_features] #e_w
        hnet_bscalars = hnet_scalars[:, self.out_features:] #e_b

        hnet_out = hnet_wscalars * F.linear(input, self.hnet_weight)


        if self.hnet_bias is not None:
            hnet_out += hnet_bscalars * self.hnet_bias

        output += hnet_out

        return self.sigma(output)

    def init_params(self):
        # Initialize elementary parameters.
        stdv = 1. / math.sqrt(self.in_features)
        self.elem_weight.data.uniform_(-stdv, stdv)
        self.hnet_weight.data.uniform_(-stdv, stdv)
        if self.elem_bias is not None:
            self.elem_bias.data.uniform_(-stdv, stdv)
            self.hnet_bias.data.uniform_(-stdv, stdv)
        # Intialize hypernet parameters.
        self.htensor_to_scalars.weight.data.normal_(std=0.01)
