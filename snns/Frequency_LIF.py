import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft

import numpy as np

from snns import base
from utils.utils import Series_decomp


class cauchy_spike_generation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, alpha=0.5):
        ctx.save_for_backward(v, torch.tensor(alpha, dtype=v.dtype, device=v.device))
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, alpha = ctx.saved_tensors
        surrogate = alpha / (np.pi * (alpha ** 2 + v ** 2))

        return grad_output * surrogate, None

class SFDAM(nn.Module):

    def __init__(self, configs):
        super(SFDAM, self).__init__()

        self.grad_history = []

        self.series_decomp = Series_decomp()
        self.use_high_freq_balance = False
        self.trend_factor = nn.Parameter(torch.tensor(1 - configs.frequency_factor).float())
        self.season_factor = nn.Parameter(torch.tensor(1 + configs.frequency_factor).float())

    def learn_high_freq(self, target=1.0, tolerance=0.1, patience=1):
        if len(self.grad_history) > patience:
            self.grad_history = self.grad_history[-patience:]

        swith_list = [r > target - tolerance for r in self.grad_history]
        if sum(swith_list) == patience:
            return True

        return False

    def switch(self, model, low_freq_cutoff=0.5, high_freq_cutoff=0.5):
        grads = []

        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_np = param.grad.detach().cpu().numpy()

                grad_fft = np.abs(fft(grad_np, axis=1))
                avg_spectrum = grad_fft.mean(axis=0)

                n_freqs = len(avg_spectrum)
                if n_freqs > 1:
                    low_idx = int(n_freqs * low_freq_cutoff)
                    high_idx = int(n_freqs * (1 - high_freq_cutoff))

                    low_mag = avg_spectrum[:low_idx].mean() if low_idx > 0 else avg_spectrum[0]
                    high_mag = avg_spectrum[high_idx:].mean() if high_idx < n_freqs else avg_spectrum[-1]

                    grad = low_mag / (high_mag + 1e-8)
                    grads.append(grad)

        avg_grad = np.mean(grad) if grads else 1.0
        self.grad_history.append(avg_grad)
        learn_high_freq_flag = self.learn_high_freq()

        self.use_high_freq_balance = learn_high_freq_flag

    def TS_loss(self, outputs, batch_y):
        out_season, out_trend = self.series_decomp(outputs)
        y_season, y_trend = self.series_decomp(batch_y)

        if self.use_high_freq_balance == True:
            trend_loss = F.mse_loss(out_trend, y_trend)
            seasonal_loss = F.mse_loss(out_season, y_season)
            loss = self.season_factor * seasonal_loss + self.trend_factor * trend_loss
        else:
            loss = F.mse_loss(outputs, batch_y)

        return loss


class Frequency_LIF(base.MemoryModule):
    def __init__(self, args, v_threshold=1.0, surrogate_function=cauchy_spike_generation.apply, hard_reset=False, detach_reset=False):
        super().__init__()
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.register_memory('v', 0.)
        self.register_memory('t', 0)
        self.v_threshold = nn.Parameter(torch.tensor(v_threshold).float())

        self.alpha, self.beta = args.alpha, args.beta
        self.beta_x = nn.Parameter(torch.tensor(1.0).float())
        self.min_beta = args.min_beta
        self.beta_decay_factor = args.beta_decay_factor

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
        cond = self.v - self.v_threshold
        spike = self.surrogate_function(cond, self.alpha)

        return spike

    def beta_decay_exponential(self):
        decay_factor = (self.min_beta / self.beta) ** (self.beta_decay_factor)
        res_beta = max(self.beta * (decay_factor ** self.t), self.min_beta)

        return res_beta

    def neuronal_charge(self, x: torch.Tensor):
        beta = self.beta_decay_exponential()
        new_x = beta * self.v + self.beta_x * x
        self.v = new_x

    def TFDAM(self, x):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        self.t += 1

        return spike

    def v_float_to_tensor(self, x):
        if isinstance(self.v, float):
            self.v = torch.full_like(x, fill_value=0)

    def forward(self, x):
        self.v_float_to_tensor(x)
        spike = self.TFDAM(x)

        return spike



