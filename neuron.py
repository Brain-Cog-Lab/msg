import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from surrogate import *
from copy import deepcopy


class BaseNeuron(nn.Module):
    def __init__(self,
                 act_func,
                 threshold: float = 1.,
                 soft_mode: bool = False,
                 **kwargs
                 ):
        super(BaseNeuron, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        self.mem = 0.
        self.spike = 0.
        self.summem = 0.
        self.sumspike = 0.
        self.record_sum = True
        self.v_reset = 0.
        self.soft_mode = soft_mode
        self.act_func = act_func

    def cal_spike(self):
        self.spike = self.act_func(self.mem - self.threshold)

    def cal_mem(self, x):
        raise NotImplementedError

    def hard_reset(self):
        self.mem = self.mem * (1 - self.spike)

    def soft_reset(self):
        self.mem = self.mem - self.threshold * self.spike.detach()

    def forward(self, x):
        self.cal_mem(x)
        self.cal_spike()
        if self.record_sum:
            self.summem += x.detach()
            self.sumspike += self.spike.detach()
        self.soft_reset() if self.soft_mode else self.hard_reset()

        return self.spike

    def reset(self):
        self.mem = self.v_reset
        self.spike = 0.

        self.summem = 0.
        self.sumspike = 0.

    def set_threshold(self, threshold):
        self.threshold = Parameter(torch.tensor(threshold, dtype=torch.float), requires_grad=False)

    def set_tau(self, tau):
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError


class IF(BaseNeuron):
    def __init__(self, act_func, threshold=1., **kwargs):
        super().__init__(act_func, threshold, **kwargs)

    def cal_mem(self, x):
        self.mem = self.mem + x


class LIF(BaseNeuron):
    def __init__(self, act_func, threshold=1., **kwargs):
        super().__init__(act_func, threshold, **kwargs)
        self.tau = kwargs['tau']

    def cal_mem(self, x):
        # self.mem = self.mem * (1 - 1. / self.tau) + x
        self.mem = self.mem + (x - self.mem) / self.tau




