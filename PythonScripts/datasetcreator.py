import os

import numpy as np
import pandas
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from skimage import io

from PIL import Image
import os, sys



class WatermelonDataset(Dataset):
    def __init__(self, csv_file, root_dir_imgs, transform=None):
        self.annotations = pandas.read_csv(csv_file)
        self.root_dir_imgs = root_dir_imgs
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # get specific image and corresponding target

        # iloc opens up csv file, we are finding image path of item in row index, column 0

        # joining path --> /parent + /child
        img_path = os.path.join(self.root_dir_imgs, self.annotations.iloc[index, 0])
        image = io.imread(img_path)


        taste_value = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return image , taste_value


