import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.xavier_uniform_(m.bias)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(192 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # self.eval_inner = eval_inner

    def forward(self, x):
        # layers_out = []
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.dropout(x, training=self.training, p=0.1)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.dropout(x, training=self.training, p=0.1)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.dropout(x, training=self.training, p=0.1)
        x = x.view(-1, self.num_flat_features(x))
        # if not self.training:
        #     layers_out.append(x)
        x = F.relu(self.fc1(x))
        # if not self.training:
        #     layers_out.append(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.fc2(x))
        # if not self.training:
        #     layers_out.append(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.fc3(x)
        return x
        # if not self.training:
        #     layers_out.append(x)
        # layers_out.reverse()
        # if self.eval_inner:
        #     return F.softmax(x, dim=1), layers_out
        # else:
        #     return F.softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

