from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class FashionMNIST(Dataset):
    """EMNIST dataset."""

    def __init__(self, root, train, transform=None, target_transform=None):

        self.dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=True,
                                                         transform=transform, target_transform=target_transform)
        self.tt = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {'data': image, 'label': label}
        return sample

    def getclasses(self):
        classes = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
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
