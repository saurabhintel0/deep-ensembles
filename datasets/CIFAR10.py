from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

# normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
# transforms = \
#     {'normalize': transforms.Compose([transforms.ToTensor(), normalize]),
#      'augment': transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(), normalize])}

class CIFAR10(Dataset):
    """CIFAR10 dataset."""

    # def __init__(self, root, train, transform_type='normalize', target_transform=None):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform,
                                                    download=True, target_transform=target_transform)
        self.transform = transform
        self.tt = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {'data': image, 'label': label}
        return sample

    def getclasses(self):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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
