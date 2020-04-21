'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class ICLeNet(nn.Module):
    def __init__(self, p0, p1, p2, ICL):
        super(ICLeNet, self).__init__()
        self.ICL = ICL

        ### Chunk 1
        if self.ICL:
            self.batchNormIC1 = nn.BatchNorm2d(3) # unknown size currently...
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        ### Chunk 2
        if self.ICL:
            self.batchNormIC2 = nn.BatchNorm2d(6) # unknown size currently...
        
        self.conv2 = nn.Conv2d(6, 16, 5)

        ### Chunk 3
        if self.ICL:
            self.batchNormIC3 = nn.BatchNorm2d(16) # unknown size currently...
        
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.batch_size = 0
        self.p0 = p0; # dropout at input
        self.p1 = p1;
        self.p2 = p2;

    def forward(self, x):
        orig = x
        self.batch_size = x.shape[0]
        h, w = x.shape[2:]
        print(x.shape)
        # Chunk 1
        if self.ICL:
            # x = x.view(self.batch_size, -1)
            out = F.dropout(self.batchNormIC1(x), p=self.p0, training=self.training)
            # out = out.view(self.batch_size, 32, h, w)
        else:
            out = x

        out = F.relu(self.conv1(out))
        activations = out
        out = F.max_pool2d(out, 2)

        # Chunk 2
        if self.ICL:
            # out = out.view(self.batch_size, -1)
            out = F.dropout(self.batchNormIC2(out), p=self.p1, training=self.training)
            # out = out.view(self.batch_size, 32, h, w)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        
        # Chunk 3
        if self.ICL:
            # out = out.view(self.batch_size, -1)
            out = F.dropout(self.batchNormIC3(out), p=self.p2, training=self.training)
            # out = out.view(self.batch_size, 32, h, w)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        
        out = self.fc3(out)
        return out, orig, activations
