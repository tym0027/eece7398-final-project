import torch
import torch.nn as nn
import torch.nn.functional as F
'''
import os
import sys
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import torch.optim
from torch.optim import lr_scheduler

import random

import cv2
import skimage.io

from datetime import datetime
'''

# Random Seed
class ICTimNet(nn.Module):
    def __init__(self, p0, p1, p2, p3, ICL):
        super(ICTimNet, self).__init__()
        # My code
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.batch_size = 0
        self.ICL = ICL

        ### CHUNK 1
        # IC - Layer
        if self.ICL:
            self.batchNormIC1 = nn.BatchNorm2d(3) # unknown size currently... originally 32768
        
        self.conv1a = nn.Conv2d(3, 32, (5,5), padding=2, stride=1)

        ### CHUNK 2
        # IC - Layer
        if self.ICL:
            self.batchNormIC2 = nn.BatchNorm2d(32) # unknown size currently... 32768

        self.conv2a = nn.Conv2d(32, 32, (5,5), padding=2, stride=1)
        self.conv2b = nn.Conv2d(32, 32, (5,5), padding=2, stride=1)
        self.conv2c = nn.Conv2d(32, 16, (3,3), padding=1, stride=1)
        self.conv2d = nn.Conv2d(16, 8, (3,3), padding=1, stride=1)
        
        ### CHUNK 3
        # IC - Layer
        if self.ICL:
            self.batchNormIC3  = nn.BatchNorm2d(8) # unknown size currently...
        
        self.conv3a = nn.Conv2d(8, 8, (3,3), padding=1, stride=1)
        self.conv3b = nn.Conv2d(8, 4, (3,3), padding=1, stride=1)
        
        ### CHUNK 4
        # IC - Layer
        if self.ICL:
            self.batchNormIC4 = nn.BatchNorm2d(4) # unknown size currently... 
        
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024) # 3072 -> 2048
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 512)
        
        self.linear5 = nn.Linear(512, 10)
        

    def forward(self, x, p0, p1, p2, p3):
        # My code
        orig = x
        self.batch_size = x.shape[0]
        h, w = x.shape[2:]
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        ### Chunk 1
        # IC Layer
        if self.ICL:
            x = F.dropout(self.batchNormIC1(x), p=self.p0, training=self.training)

        x = F.relu(self.conv1a(x))
        activations = x


        ### Chunk 2
        # IC Layer
        if self.ICL:
            x = F.dropout(self.batchNormIC2(x), p=self.p1, training=self.training)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.relu(self.conv2c(x))
        x = F.relu(self.conv2d(x))

        ### Chunk 3
        # IC Layer
        if self.ICL:
            x = F.dropout(self.batchNormIC3(x), p=self.p2, training=self.training)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))

        ### CHUNK 4
        # IC - Layer
        if self.ICL:
            x = F.dropout(self.batchNormIC4(x), p=self.p3, training=self.training)
        
        x = x.view(self.batch_size, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)

        return x, orig, activations
