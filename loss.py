import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceScore(nn.Module):
    def __init__(self, smooth=1.0, axis=None, threshold=0.5):
        super(DiceScore, self).__init__()
        self.smooth = smooth
        self.axis = axis
        self.threshold = threshold

    def forward(self, inputs, targets):  
        inputs = F.softmax(inputs, dim=1)
        inputs = (inputs > self.threshold).float()
        if self.axis:
            inputs = inputs[:, self.axis]
            targets = targets[:, self.axis]
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, axis=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.axis = axis

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        # inputs = F.log_softmax(inputs, dim=1)
        if self.axis:
            inputs = inputs[:, self.axis]
            targets = targets[:, self.axis]
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        return 1 - dice


class DiceCELoss(nn.Module):
    def __init__(self, binary=True, smooth=1.0, axis=None):
        super(DiceCELoss, self).__init__()
        self.binary = binary
        if axis:
            self.dice = [DiceLoss(smooth=smooth, axis=i) for i in axis] 
        else:
            self.dice = [DiceLoss(smooth=smooth)]

    def forward(self, inputs, targets): 
        # dice_loss = 0.0
        # coeff = 1.
        # for _dice in self.dice:
        #     dice_loss += coeff * _dice(inputs, targets) 
        #     coeff *= 2
        dice_loss = self.dice[0](inputs, targets) + 2*self.dice[1](inputs, targets)
        # if self.binary:
        #     ce = F.binary_cross_entropy(inputs, targets)
        # else:
        #     targets = targets.argmax(1).long()
        #     # targets = targets.type(torch.cuda.LongTensor) if torch.cuda.is_available() else targets.type(torch.LongTensor)
        #     ce = F.cross_entropy(inputs, targets)
        Dice_CE = dice_loss
        return Dice_CE


class FocalLoss(nn.Module):
    def __init__(self, binary=True, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.binary = binary
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.binary:
            ce = F.binary_cross_entropy(inputs, targets)
        else:
            targets = targets.argmax(1).long()
            # targets = targets.type(torch.cuda.LongTensor) if torch.cuda.is_available() else targets.type(torch.LongTensor)
            ce = F.nll_loss(inputs, targets)
        ce_exp = torch.exp(-ce)
        focal_loss = self.alpha * (1-ce_exp)**self.gamma * ce            
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.75):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        tp = (inputs * targets).sum()    
        fp = ((1-targets) * inputs).sum()
        fn = (targets * (1-inputs)).sum()
       
        Tversky = (tp + smooth) / (tp + self.alpha*fn + (1-self.alpha)*fp + self.smooth)  
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.75, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        tp = (inputs * targets).sum()    
        fp = ((1-targets) * inputs).sum()
        fn = (targets * (1-inputs)).sum()
        
        Tversky = (tp + self.smooth) / (tp + self.alpha*fn + (1-self.alpha)*fp + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma 
        return FocalTversky