from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
# from skimage import io
import cv2
from PIL import Image
import numpy as np

normalize_mnist = transforms.Normalize([0.13], [0.3081])
normalize_cifar = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

transforms = {'normalize_mnist': transforms.Compose([transforms.ToTensor(), normalize_mnist]),
              'normalize_cifar': transforms.Compose([transforms.ToTensor(), normalize_cifar]),
              'augment_cifar': transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(), normalize_cifar])}


class ImConcatDataset(Dataset):
    """Generic class for adversarial dataset."""

    def __init__(self, csv_file1, root_dir1, csv_file2, root_dir2, transform_type=None):
        data_frame1 = pd.read_csv(csv_file1)
        self.root_dir1 = root_dir1
        self.len1 = len(data_frame1)
        data_frame2 = pd.read_csv(csv_file2)
        self.root_dir2 = root_dir2
        self.len2 = len(data_frame2)
        self.data_frame = pd.concat([data_frame1, data_frame2])
        self.transform = transforms[transform_type]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx < self.len1:
            img_name = os.path.join(self.root_dir1, self.data_frame.iloc[idx, 0])
        else:
            img_name = os.path.join(self.root_dir2, self.data_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        image = Image.open(img_name)
        # if image.ndim == 2:
        #     image = np.expand_dims(image, 2)
        # image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        # if image.ndim == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # elif image.ndim == 2:
        #     image = np.expand_dims(image, 2)

        # if image.dtype == 'uint16':
        #     scale_factor = 65535.0
        # else:
        #     scale_factor = 255.0
        # image = 2*(image / scale_factor - 0.5)
        # image = image.astype(np.float32)

        image = self.transform(image)
        labels = self.data_frame.iloc[idx, 1]
        sample = {'data': image, 'label': labels}
        return sample

    def getclasses(self):
        # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        classes = [str(i) for i in range(10)]
        c = dict()
        for i in range(len(classes)):
            c[i] = classes[i]
        return c


