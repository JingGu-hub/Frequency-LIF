import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Series_decomp

class TrendSeasonLoss(nn.Module):

    def __init__(self, season_factor_value=1.0):
        super(TrendSeasonLoss, self).__init__()
        self.series_decomp = Series_decomp()
        self.use_high_freq_balance = False

        self.trend_factor = nn.Parameter(torch.tensor(1 - season_factor_value).float())
        self.season_factor = nn.Parameter(torch.tensor(1 + season_factor_value).float())

    def set_use_high_freq_balance(self):
        self.use_high_freq_balance = True

    def forward(self, outputs, batch_y):
        out_season, out_trend = self.series_decomp(outputs)
        y_season, y_trend = self.series_decomp(batch_y)

        if self.use_high_freq_balance == True:
            trend_loss = F.mse_loss(out_trend, y_trend)
            seasonal_loss = F.mse_loss(out_season, y_season)
            loss = self.season_factor * seasonal_loss + self.trend_factor * trend_loss
        else:
            loss = F.mse_loss(outputs, batch_y)

        return loss
