'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, p0, p1, p2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.p0 = p0; # dropout at input
        self.p1 = p1; # dropout at input to next layer
        self.p2 = p2; # dropout activation functions

    def forward(self, x):
        # input data dropout
        out = F.dropout(x, p=self.p0)

        # dropping activation output
        out = F.dropout(F.relu(self.conv1(out)), p=self.p2)
        
        # dropping input to next layer
        out = F.dropout(out, p=self.p1)

        out = F.dropout(F.max_pool2d(out, 2), p=self.p2)
        out = F.dropout(out, p=self.p1)
        out = F.dropout(F.relu(self.conv2(out)), p=self.p2)
        out = F.dropout(out, p=self.p1)
        out = F.dropout(F.max_pool2d(out, 2), p=self.p2)
        out = F.dropout(out, p=self.p1)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.fc1(out)), p=self.p2)
        out = F.dropout(out, p=self.p1)
        out = F.dropout(F.relu(self.fc2(out)), p=self.p2)
        
        out = self.fc3(out)
        return out
