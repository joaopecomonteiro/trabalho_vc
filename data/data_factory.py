from torch.utils.data import DataLoader, RandomSampler
import torch
import numpy as np

from .dataset import SpaceDebrisDataset

def build_datasets(root_dir="debris_detection"):
    train_dataset = SpaceDebrisDataset(root_dir, "train")
    val_dataset = SpaceDebrisDataset(root_dir, "val")

    return train_dataset, val_dataset


def build_data_loaders(config):

    train_data, val_data = build_datasets(config.root_dir)

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True,
                              num_workers=2)


    val_loader = DataLoader(val_data, config.batch_size, shuffle=True,
                              num_workers=2)


    return train_loader, val_loader











