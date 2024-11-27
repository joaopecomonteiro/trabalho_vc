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

from losses import n_debris_Loss


class HeadNetwork(nn.Module):

    def __init__(self, in_channels ,n_debris):
        super().__init__()

        self.n_debris = n_debris
        if self.n_debris == 0:
            self.n_debris = 2
        #self.conv = nn.Conv2d(in_channels, self.n_debris, kernel_size=3, stride=1, padding=1)
        #self.flatten = nn.Flatten()
        print(in_channels)
        self.lin1 = nn.Linear(in_channels, int(in_channels/2))
        self.lin2 = nn.Linear(int(in_channels/2), int(in_channels/4))
        self.lin3 = nn.Linear(int(in_channels/4), self.n_debris*4)

    def forward(self, features):


        #breakpoint()

        if self.training:
            #breakpoint()
            #drop_features = F.dropout(features, p=0.5, training=True)
            #feat = self.conv(drop_features)
            #flat = nn.Flatten()(drop_features)
            lin1_feats = self.lin1(features)
            lin2_feats = self.lin2(lin1_feats)
            logits = self.lin3(lin2_feats)


            #logits = self.lin(flat)
            #breakpoint()
            return logits


class PyramidNetwork(nn.Module):

    def __init__(self, frontend, n_debris_classifier):
        super().__init__()

        self.frontend = frontend
        self.n_debris_classifier = n_debris_classifier

        #self.conv1 = nn.Conv2d(256*3, 256*2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.lin1 = nn.Linear(32768, 16384)
        self.lin2 = nn.Linear(16384, 8192)
        self.lin3 = nn.Linear(8192, 4096)
        self.heads = []
        self.n_debris_found = []

    def forward(self, image, *args):


        feature_maps = self.frontend(image)

        n_debris = self.n_debris_classifier(feature_maps)

        rounded_n_debris = int(torch.round(n_debris))

        #size = (feature_maps[1].shape[-1], feature_maps[1].shape[-1])
        #upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        #up_features = []
        #for feature in feature_maps[1:-2]:
        #    up_features.append(upsample(feature))

        #conc_features = torch.cat(up_features, dim=1)
        conc_features = feature_maps[2]
        #breakpoint()
        if rounded_n_debris not in self.n_debris_found:
            #bisect.insort(self.n_debris_found, rounded_n_debris)
            self.n_debris_found.append(rounded_n_debris)
            index = self.n_debris_found.index(rounded_n_debris)

            #breakpoint()
            head = HeadNetwork(4096,  rounded_n_debris)
            self.heads.append(head)
        else:
            index = self.n_debris_found.index(rounded_n_debris)
            #head = self.heads[index]
        #breakpoint()
        conv1_feats = self.conv1(conc_features)
        #conv2_feats = self.conv2(conv1_feats)

        drop_features = F.dropout(conv1_feats, p=0.5, training=True)
        flat_feats = nn.Flatten()(drop_features)
        lin1_feats = self.lin1(flat_feats)
        lin2_feats = self.lin2(lin1_feats)
        lin3_feats = self.lin3(lin2_feats)
        #lin1 = self.lin1(flat_feats)

        logits = self.heads[index](lin3_feats)

        bboxes = logits.reshape(n_debris, 4)

        #logits = head(conc_features)
        #breakpoint()
        return n_debris, bboxes


