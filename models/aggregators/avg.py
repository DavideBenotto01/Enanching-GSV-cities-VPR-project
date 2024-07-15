import torch
import torch.nn.functional as F
import torch.nn as nn

class AvgPool(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps), (x.size(-2), x.size(-1)))
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)
    
    
    