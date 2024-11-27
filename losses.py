import torch
import torch.nn.functional as F
import torch.nn as nn


def n_debris_Loss(n_debris_prediction, bboxes):
    #breakpoint()
    n_gt = torch.tensor(bboxes.shape[1])
    n_debris_prediction = n_debris_prediction.view(-1).float()
    n_gt = n_gt.view(-1).float()
    return F.mse_loss(n_debris_prediction, n_gt)

def find_pairs(pred_bboxes, gt_bboxes, n_debris):
    distances = torch.cdist(pred_bboxes, gt_bboxes, p=2)

    paired_indices = []
    used_rows = set()
    used_cols = set()

    for _ in range(n_debris):
        # Find the minimum distance and corresponding indices
        min_val, min_idx = torch.min(distances, dim=1)  # Min per row
        row_idx = torch.argmin(min_val).item()
        col_idx = min_idx[row_idx].item()

        # Record the pair
        paired_indices.append((row_idx, col_idx))
        used_rows.add(row_idx)
        used_cols.add(col_idx)

        # Mask out the used row and column to avoid reuse
        distances[row_idx, :] = float('inf')
        distances[:, col_idx] = float('inf')



def bboxes_loss(pred_bboxes, gt_bboxes):
    pass



class N_Debris_Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n_debris_prediction, bboxes):
        return n_debris_Loss(n_debris_prediction, bboxes)


