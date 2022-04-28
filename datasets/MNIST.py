from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class MNIST(Dataset):
    """MNIST dataset."""

    def __init__(self, root, train, transform=None, target_transform=None):
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, transform=transform,
                                                  download=True, target_transform=target_transform)
        self.tt = target_transform
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {'data': image, 'label': label}
        return sample

    def getclasses(self):
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
                # print(self.tt(i))
                # c.add(classes[self.tt(i)])
        return c

    def get_transform(self):
        return self.transform

