import os
import torch
import torch.nn.functional as F

from torch import nn

os.environ["CUDA_VISIVLE_DEVICES"] = '1' 

class MLP(nn.Module):
    # q(h) = h + sigma(h) \odot \epsilon
    def __init__(self, input_size, device):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc1.weight = nn.Parameter(torch.zeros(input_size, input_size))
        self.fc1.bias = nn.Parameter(torch.zeros(input_size))
        self.device = device
        
    def forward(self, x):
        epsilon = torch.empty(x.size()).normal_(mean=0, std=1).cuda()
        output = self.fc1(x)
        output = output * epsilon + x
        return output
