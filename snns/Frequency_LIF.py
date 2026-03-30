import numpy as np
import torch
from snns import base
import torch.nn as nn

class Frequency_LIF(base.MemoryModule):
    def __init__(self, alpha=0.1, use_dynamic_beta=True, beta=0.7, beta_decay_factor=0.5,
                 v_threshold=1.0, surrogate_function=None, hard_reset=False, detach_reset=False):
        super().__init__()
        self.alpha = alpha

        self.use_dynamic_beta = use_dynamic_beta
        self.beta_x = nn.Parameter(torch.tensor(1.0).float())
        self.beta, self.beta_decay_factor = beta, beta_decay_factor

        self.register_memory('v', 0.)
        self.v_threshold = nn.Parameter(torch.tensor(v_threshold).float())

        self.hard_reset = hard_reset
        self.detach_reset = detach_reset

        self.surrogate_function = surrogate_function

    def beta_decay_exponential(self, t=0, min_beta=0.5):
        decay_factor = (min_beta / self.beta) ** (self.beta_decay_factor)  # 约0.9036
        res_beta = max(self.beta * (decay_factor ** t), min_beta)

        return res_beta

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.hard_reset:
            self.v = self.v * (1. - spike_d)
        else:
            self.v = self.v - spike_d * self.v_threshold

    def neuronal_fire(self):
        alpha = self.alpha

        # 保持连续值，但实现"或"的逻辑
        cond = self.v - self.v_threshold
        spike = self.surrogate_function(cond, alpha)

        return spike

    def neuronal_charge(self, x: torch.Tensor, t):
        beta = self.beta_decay_exponential(t=t) if self.use_dynamic_beta else self.beta
        new_x = beta * self.v + self.beta_x * x
        self.v = new_x

    def v_float_to_tensor(self, x):
        if isinstance(self.v, float):
            self.v = torch.full_like(x, fill_value=0)

    def set_learn_high_freq(self):
        self.use_learn_high_freq = 1.0

    def forward(self, x, t=0):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x, t)
        spike = self.neuronal_fire()

        self.neuronal_reset(spike)

        return spike



