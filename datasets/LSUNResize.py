from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
# from skimage import io
# import cv2
import numpy as np
from PIL import Image


class LSUNResize(Dataset):
    """LSUN (Resized) dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tt = target_transform
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        labels = self.data_frame.iloc[idx, 1]
        if self.tt:
            labels = self.tt(labels)
        sample = {'data': image, 'label': labels}
        return sample

    def getclasses(self):
        classes = ['0']
        c = dict()
        if self.tt is None:
            for i in range(len(classes)):
                c[i] = classes[i]
        else:
            for i in range(len(classes)):
                k = self.tt(i)
                if k in c:
                    c[k].append(classes[i])
                else:
                    c[k] = [classes[i]]
        return c

    def get_transform(self):
        return self.transform

