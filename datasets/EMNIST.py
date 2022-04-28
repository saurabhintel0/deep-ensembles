from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class EMNIST(Dataset):
    """EMNIST dataset."""

    classes = None

    def __init__(self, root, split, train, transform=None, target_transform=None):

        self.dataset = torchvision.datasets.EMNIST(root=root, split=split, train=train, target_transform=target_transform,
                                                   transform=transform, download=True)
        self.split = split
        self.tt = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {'data': image, 'label': label}
        return sample

    def getclasses(self):
        classes = None
        if self.split == 'letters':
            classes = [chr(x) for x in range(ord('A'), ord('Z')+1)]
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

