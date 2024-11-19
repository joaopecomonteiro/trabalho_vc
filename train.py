import math
from operator import mul
from functools import reduce
import torch.nn as nn
import torch
from argparse import ArgumentParser
from yacs.config import CfgNode
import os


from pyramid import PyramidNetwork
from FPN import FPN50
from data.data_factory import build_data_loaders

def load_config(config_path):
    with open(config_path) as f:
        return CfgNode.load_cfg(f)


def get_configuration():
    # Load config defaults
    #config = get_default_configuration()

    defaults_path = "config.yml"
    config = load_config(defaults_path)
    # Finalise config
    config.freeze()

    return config


def build_model(config):

    fronted = FPN50()

    return PyramidNetwork(fronted)




def train(dataloader, model, config, epoch):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iteration = (epoch - 1) * len(dataloader)

    for i, batch in enumerate(dataloader):

        #if len(config.gpus) > 0:
        batch = [t.to(device) for t in batch]

        images, bboxes = batch

        logits = model(images)

        breakpoint()


def main():

    config = load_config("config.yml")

    #breakpoint()

    model = build_model(config)
    train_loader, val_loader = build_data_loaders(config)

    epoch = 0

    while epoch <= config.num_epochs:
        print('\n\n=== Beginning epoch {} of {} ==='.format(epoch,
                                                            config.num_epochs))

        train(train_loader, model, config, epoch)


if __name__ == '__main__':
    main()

