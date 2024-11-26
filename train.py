import math
from operator import mul
from functools import reduce
import torch.nn as nn
from torch.optim import SGD
import torch
from argparse import ArgumentParser
from yacs.config import CfgNode
import os
from tqdm import tqdm


from data.data_factory import build_data_loaders
from model_factory import build_model, build_criterion

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







def train(dataloader, model, criterion, optimiser, config, epoch):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iteration = (epoch - 1) * len(dataloader)

    for i, batch in tqdm(enumerate(dataloader)):

        #breakpoint()
        #if len(config.gpus) > 0:
        batch = [t.to(device) for t in batch]

        images, bboxes = batch

        logits = model(images)

        loss = criterion(logits, bboxes)
        #for i in range(len(logits)):
        #    print(logits[i].shape)
        #print(loss)

        #breakpoint()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #breakpoint()
        iteration += 1

        if i % 50 == 0:
            print(f"i:{i}, loss:{torch.sqrt(loss)}")


def evaluate(dataloader, model, criterion, config, epoch):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_loss = 0
    for i, batch in enumerate(dataloader):

            # Move tensors to GPU
            batch = [t.to(device) for t in batch]
            images, bboxes = batch
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, bboxes)
            mean_loss += loss
    mean_loss /= len(dataloader)
    return mean_loss

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_iou):
    if isinstance(model, nn.DataParallel):
        model = model.module

    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_iou': best_iou
    }

    torch.save(ckpt, path)


def main():

    config = load_config("config.yml")

    #breakpoint()

    model = build_model(config)
    criterion = build_criterion(config)
    train_loader, val_loader = build_data_loaders(config)

    optimiser = SGD(model.parameters(), config.learning_rate, weight_decay=config.weight_decay)

    epoch = 0

    while epoch <= config.num_epochs:
        print('\n\n=== Beginning epoch {} of {} ==='.format(epoch,
                                                            config.num_epochs))

        train(train_loader, model, criterion, optimiser,config, epoch)

        val_mse = evaluate(val_loader, model, criterion, config, epoch)
        print(f"val_mse: {val_mse}")

        epoch += 1

if __name__ == '__main__':
    main()

