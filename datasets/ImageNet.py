from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import os


class ImageNet(Dataset):
    """ImageNet dataset."""

    def __init__(self, root, split, transform=None, target_transform=None):
        folder = os.path.join(root, split)
        self.dataset = torchvision.datasets.ImageFolder(folder, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.tt = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {'data': image, 'label': label}
        return sample

    def getclasses(self):
        # classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        classes = [str(i) for i in range(1000)]
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

