from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
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
                              num_workers=2, collate_fn=custom_collate)


    val_loader = DataLoader(val_data, config.batch_size, shuffle=True,
                              num_workers=2, collate_fn=custom_collate)


    return train_loader, val_loader



def custom_collate(batch):
    # Extract individual elements from the batch
    images, bboxes = zip(*batch)

    # Pad cam_images to ensure they have the same shape within a batch
    #padded_cam_images = pad_sequence(cam_images, batch_first=True, padding_value=0)

    padded_bboxes = pad_sequence(bboxes, batch_first=True, padding_value=0)

    # Pad cam_calibs to ensure they have the same shape within a batch
    #padded_cam_calibs = pad_sequence(cam_calibs, batch_first=True, padding_value=0)

    # Pad cam_positions to ensure they have the same shape within a batch
    #padded_cam_positions = pad_sequence(cam_positions, batch_first=True, padding_value=0)

    # Convert other elements to tensors
    #ego_position_tensor = torch.from_numpy(np.array(ego_position))
    #full_labels_tensor = torch.from_numpy(np.array(full_labels))
    #full_mask_tensor = torch.from_numpy(np.array(full_mask))

    bboxes = torch.from_numpy(np.array(padded_bboxes))

    images = torch.from_numpy(np.array(images))

    return images, bboxes










