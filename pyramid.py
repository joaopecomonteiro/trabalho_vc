import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
import pickle


class PyramidNetwork(nn.Module):

    def __init__(self, frontend, n_debris_classifier):
        super().__init__()

        self.frontend = frontend
        self.n_debris_classifier = n_debris_classifier
        #self.transformer = transformer
        #self.topdown = topdown
        #self.classifier = classifier

    def forward(self, image, *args):
        # Extract multiscale feature maps
        #breakpoint()

        feature_maps = self.frontend(image)

        n_debris = self.n_debris_classifier(feature_maps)

        # Transform image features to birds-eye-view
        #bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        #td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        #logits = self.classifier(td_feats)

        return n_debris


