import math
import numpy as np
import torch

class SpikeAct_extended(torch.autograd.Function):
    '''
    solving the non-differentiable term of the Heavisde function
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu

class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

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

class Rectangle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        spikes = input.ge(0.).float()
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5

        return grad_input * temp.float()

@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    # grad_input = grad_output.clone()
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class SpikeGeneration(torch.autograd.Function):
    """
    脉冲生成函数（包含替代梯度）
    """

    @staticmethod
    def forward(ctx, v, alpha=0.5):
        ctx.save_for_backward(v, torch.tensor(alpha, dtype=v.dtype, device=v.device))
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, alpha = ctx.saved_tensors
        surrogate = alpha / (np.pi * (alpha ** 2 + v ** 2))

        return grad_output * surrogate, None

