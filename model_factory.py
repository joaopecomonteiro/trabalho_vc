from pyramid import PyramidNetwork
from FPN import FPN50, FPN101
from N_Debris import N_debris_classifier
from losses import N_Debris_Criterion


def build_model(config):

    fpn = FPN50()
    ndc = N_debris_classifier(256)
    return PyramidNetwork(fpn, ndc)


def build_criterion(config):
    criterion = N_Debris_Criterion()
    return criterion