import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import ast
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor


class SpaceDebrisDataset(Dataset):
    """
    Dataset
    """

    def __init__(self, root_dir = "debris_detection", split="train"):

        self.root_dir=root_dir
        self.split=split

        self.split_dir = os.path.join(self.root_dir,
                                 f"{self.split}")

        self.df = pd.read_csv(f"{self.split_dir}.csv")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        camera_path = f"{self.split_dir}/{idx}.jpg"

        image = to_tensor(Image.open(camera_path))

        bboxes_df = self.df[self.df["ImageID"] == idx]

        bboxes_temp = bboxes_df["bboxes"]

        bboxes_text = np.array(bboxes_temp)[0]

        bboxes = torch.tensor(ast.literal_eval(bboxes_text))

        return image, bboxes







if __name__ == "__main__":
    dataset = SpaceDebrisDataset(split="train")


    idx = np.random.randint(len(dataset))

    img, bboxes = dataset[idx]

    plt.imshow(img)
    plt.show()

    #breakpoint()





























