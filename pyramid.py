import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fontTools.tfmLib import HEADER_SIZE1
from tqdm import tqdm
from time import time
import pickle
import bisect

class HeadNetwork(nn.Module):

    def __init__(self, in_channels ,n_debris):
        super().__init__()

        self.n_debris = n_debris
        if self.n_debris == 0:
            self.n_debris = 1
        self.conv = nn.Conv2d(in_channels, self.n_debris, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, features):
        size = (features[0].shape[-1], features[0].shape[-1])
        upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        up_features = []
        for feature in features[:-2]:
            up_features.append(upsample(feature))

        conc_features = torch.cat(up_features, dim=1)

        breakpoint()

        if self.training:
            #breakpoint()
            drop_features = F.dropout(conc_features, p=0.5, training=True)
            feat = self.conv(drop_features)
            flat = self.flatten(feat)

            breakpoint()



class PyramidNetwork(nn.Module):

    def __init__(self, frontend, n_debris_classifier):
        super().__init__()

        self.frontend = frontend
        self.n_debris_classifier = n_debris_classifier

        self.heads = []
        self.n_debris_found = []

    def forward(self, image, *args):


        feature_maps = self.frontend(image)

        n_debris = self.n_debris_classifier(feature_maps)

        rounded_n_debris = int(torch.round(n_debris))

        #if rounded_n_debris == 0:
        #    return n_debris,

        if rounded_n_debris not in self.n_debris_found:
            #bisect.insort(self.n_debris_found, rounded_n_debris)
            self.n_debris_found.append(rounded_n_debris)
            index = self.n_debris_found.index(rounded_n_debris)

            breakpoint()
            head = HeadNetwork(feature_maps[0].shape[1]*3, rounded_n_debris)
            self.heads.append(head)
        else:
            index = self.n_debris_found.index(rounded_n_debris)
            head = self.heads[index]

        logits = head(feature_maps)
        breakpoint()
        return n_debris


