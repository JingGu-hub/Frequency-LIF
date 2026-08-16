from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn

from snns import base

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

gamma = 0.5  # gradient scale
lens = 0.5
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = 6.0
        hight = 0.15
        temp = (gaussian(input, 0., lens) * (1. + hight)
                - gaussian(input, lens, scale * lens) * hight
                - gaussian(input, -lens, scale * lens) * hight)
        return grad_output * temp * gamma

class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 surrogate_function=None,
                 hard_reset: bool = False,
                 detach_reset: bool = False):

        assert isinstance(v_threshold, float)
        assert isinstance(hard_reset, bool)
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)

        # self.v_threshold = nn.Parameter(torch.tensor(1.0))
        self.v_threshold = v_threshold

        self.hard_reset = hard_reset
        self.detach_reset = detach_reset

        self.surrogate_function = surrogate_function

    def forward(self, x: torch.Tensor, t=0):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()

        self.neuronal_reset(spike)
        return spike

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.hard_reset:
            self.v = self.v * (1. - spike_d)
        else:
            self.v = self.v - spike_d * self.v_threshold

    def reset_zeros(self):
        self.v = None

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, detach_reset={self.detach_reset}, hard_reset={self.hard_reset}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float) or self.v is None:
            self.v = torch.full_like(x, fill_value=0)


class LIFNode(BaseNode):
    def __init__(self,
                 v_threshold: float = 1.,
                 surrogate_function: Callable = ActFun_adp.apply,
                 hard_reset: bool = False,
                 detach_reset: bool = False):
        self.decay_factor = torch.tensor(0.8).float()

        super().__init__(v_threshold, surrogate_function, hard_reset, detach_reset)

    def extra_repr(self):
        return super().extra_repr() + f', decay_factor={self.decay_factor}'

    def neuronal_charge(self, x: torch.Tensor):
        v1_new = self.v * self.decay_factor + x

        self.v = v1_new


