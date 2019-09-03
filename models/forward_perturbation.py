import torch
from torch.autograd import Function
import torch.nn as nn


class PerturbationModule(nn.Module):
    def __init__(self, T):
        super(PerturbationModule, self).__init__()
        self.T = T
        self.training = False
        self.conv_block = None

    def forward(self, x):
        if not self.training:
            x = x + self.T * torch.normal(torch.zeros_like(x), 1.0).cuda()
        return x


class PerturbationUnit(Function):
    
    
    @staticmethod
    def forward(ctx, input, T):
        ctx.T = T
        return input + ctx.T * torch.normal(torch.zeros(1), 1.0).cuda()

    @staticmethod
    def backward(ctx, grad_input):
        return grad_input, None
