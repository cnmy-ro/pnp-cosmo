from itertools import combinations
import torch
from torch import nn
import torch.nn.functional as F



class GANLoss(nn.Module):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, pred, is_real=True):
        
        if isinstance(pred, list) or isinstance(pred, tuple):
            pred = [p.flatten() for p in pred]
            pred = torch.cat(pred, dim=0)

        if self.function == 'nsgan':
            if is_real: target = torch.ones_like(pred)
            else:       target = torch.zeros_like(pred)
            F.binary_cross_entropy_with_logits(pred, target)
        
        elif self.function == 'lsgan':
            if is_real: loss = 0.5 * torch.mean((pred - 1.) ** 2)
            else:       loss = 0.5 * torch.mean(pred ** 2)

        elif self.function == 'wgan':
            if is_real: loss = -torch.mean(pred)
            else:       loss = torch.mean(pred)

        return loss


class GaussianKLLoss(nn.Module):
    """ KL loss for Gaussian distributions """

    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mu)
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1,2,3)).mean(dim=0)
        return loss



class OrthoContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content):
        num_channels = content.shape[1]        
        chan_pairs = list(combinations(list(range(num_channels)), 2))
        loss = 0
        for cp in chan_pairs:
            ci, cj = cp
            dotprod = torch.sum(content[:, ci] * content[:, cj]).abs()
            norm = torch.sum(content[:, ci]**2).sqrt() * torch.sum(content[:, cj]**2).sqrt()
            loss += dotprod / norm
        loss /= len(chan_pairs)
        return loss
