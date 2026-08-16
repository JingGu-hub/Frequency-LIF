import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
from numpy.ma.core import reshape
from sklearn.mixture import GaussianMixture


def set_seed(args):
    random.seed(args.fix_seed)
    torch.manual_seed(args.fix_seed)
    np.random.seed(args.fix_seed)

class Series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size=25, stride=1):
        super(Series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x_trend = self.moving_avg(x)
        x_season = x - x_trend

        return x_season, x_trend
