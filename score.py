import torch
import torch.nn as nn

class DiceScore(nn.Module):
    def __init__(self, eps=1e-5):
        super(DiceScore, self).__init__()

        self.eps = eps

    def forward(self, pred, label):
        overlap = torch.dot(pred.view(-1), label.view(-1))
        union = pred.sum() + label.sum()
        return 2.0*overlap/(union.float() + self.eps)
        