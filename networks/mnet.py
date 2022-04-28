import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.xavier_uniform_(m.bias)


class Net(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3, dropout=0.0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)
        im_len = (14 - 2*(kernel_size//2))//2
        self.fc1 = nn.Linear(32 * im_len * im_len, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = dropout
        # self.eval_inner = eval_inner

    def forward(self, x):
        # layers_out = []
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        return x
        # return F.softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
