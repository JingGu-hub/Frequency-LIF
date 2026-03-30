import torch
import torch.nn as nn

class FrequencyAwareLoss(nn.Module):
    def __init__(self, dataset_len=1024, threshold=1.0):
        super().__init__()
        self.base_criterion = nn.MSELoss(reduction='none')
        self.threshold = threshold
        self.dataset_len = dataset_len
        self.ratios = torch.zeros(self.dataset_len)

    def reset_ratios(self, ratios):
        self.ratios = ratios

    def forward(self, pred, target, index):
        weight = torch.where(self.ratios[index] > self.threshold, 0.1, 1.0).cuda()
        mse_loss = self.base_criterion(pred, target)

        loss = (mse_loss * weight.unsqueeze(-1).unsqueeze(-1)).mean()

        return loss


