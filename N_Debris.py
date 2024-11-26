import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

class N_debris_classifier(nn.Module):

    def __init__(self, in_channels, num_samples=40):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.num_samples = num_samples

    def initialise(self, prior):
        prior = torch.tensor(prior)
        self.conv.weight.data.zero_()
        self.conv.bias.data.copy_(torch.log(prior / (1 - prior)))

    def forward(self, features):

        #features = to_tensor(features)
        features = features[-1]
        if self.training:
            features = F.dropout2d(features, 0.5, training=True)
            feat = self.conv(features)
            flat = self.flatten(feat)
            l1 = self.linear1(flat)
            l2 = self.linear2(l1)
            l3 = self.linear3(l2)

        else:
            # At test time, apply dropout multiple times and average the result
            mean_score = 0
            for _ in range(self.num_samples):
                drop_feats = F.dropout2d(features, 0.5, training=True)
                mean_score += F.sigmoid(self.conv(drop_feats))

            mean_score = mean_score / self.num_samples

            feat = torch.log(mean_score) - torch.log1p(-mean_score)

            l1 = self.linear1(feat)
            l2 = self.linear2(l1)
            l3 = self.linear3(l2)

        return l3






