from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
# from skimage import io
# import cv2
import numpy as np
from PIL import Image



class ImDataset(Dataset):
    """Generic class for adversarial dataset."""

    def __init__(self, csv_file, root_dir, num_classes, transform=None, target_transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tt = target_transform
        self.num_classes = num_classes

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
        classes = [str(i) for i in range(self.num_classes)]
        c = dict()
        for i in range(len(classes)):
            c[i] = classes[i]
        return c

    def get_transform(self):
        return self.transform

