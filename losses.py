import torch
import torch.nn.functional as F
import torch.nn as nn


def n_debris_Loss(n_debris_prediction, bboxes):
    #breakpoint()
    n_gt = torch.tensor(bboxes.shape[1])
    n_debris_prediction = n_debris_prediction.view(-1).float()
    n_gt = n_gt.view(-1).float()
    return F.mse_loss(n_debris_prediction, n_gt)



class N_Debris_Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n_debris_prediction, bboxes):
        return n_debris_Loss(n_debris_prediction, bboxes)

