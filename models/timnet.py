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
class TimNet(nn.Module):
    def __init__(self, p0, p1, p2, BN):
        super(TimNet, self).__init__()
        # My code
        self.BN = BN
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        self.batch_size = 0

        # self.dropout_first = nn.Dropout(.1)

        self.conv1 = nn.Conv2d(3, 32, (5,5), padding=2, stride=1)
        
        if BN:
            self.batchNormP1 = nn.BatchNorm1d(32768)
        
        # self.dropout_act = nn.Dropout(.5) # originally .25

        # self.dropout_inp = nn.Dropout(.2)

        self.convc1 = nn.Conv2d(32, 32, (5,5), padding=2, stride=1)
        # self.dropout1a = nn.Dropout(.35) # originally .35
        self.convc2 = nn.Conv2d(32, 32, (5,5), padding=2, stride=1)
        # self.dropout1b = nn.Dropout(.25) # originally .25
        self.convc3 = nn.Conv2d(32, 16, (3,3), padding=1, stride=1)

        self.conv2 = nn.Conv2d(16, 8, (3,3), padding=1, stride=1)
        
        if BN:
            self.batchNormP2 = nn.BatchNorm1d(8192)
        
        # self.dropout2 = nn.Dropout(.15)
        self.conv3 = nn.Conv2d(8, 8, (3,3), padding=1, stride=1)
        if BN:
            self.batchNormP3 = nn.BatchNorm1d(8192)
        
        # self.dropout3 = nn.Dropout(.15)
        self.conv4 = nn.Conv2d(8, 4, (3,3), padding=1, stride=1)

        # self.dropout4 = nn.Dropout(.1)

        if BN:
            self.batchNorm1 = nn.BatchNorm1d(4096) # 3072 original
        
        self.linear1 = nn.Linear(4096, 2048)
        if BN:
            self.batchNorm2 = nn.BatchNorm1d(2048)
        
        self.linear2 = nn.Linear(2048, 1024) # 3072 -> 2048
        
        if BN:
            self.batchNorm3 = nn.BatchNorm1d(1024)
        
        self.linear3 = nn.Linear(1024, 1024)
        
        if BN:
            self.batchNorm4 = nn.BatchNorm1d(1024)
        
        self.linear4 = nn.Linear(1024, 512)
        
        if BN:
            self.batchNorm5 = nn.BatchNorm1d(512)
        
        self.linear5 = nn.Linear(512, 10)
        
        # self.activation = nn.ReLU()

    def forward(self, x):
        # My code
        # print("star: size: ", x.shape)
        orig = x
        # x = self.conv1(x)
        # printActivations(x, orig)
        # x = self.activation(x)

        self.batch_size = x.shape[0]

        # x = self.dropout_first(x)

        x = F.dropout(x, p=self.p0)

        x = F.dropout(F.relu(self.conv1(x)), p=self.p2)
        # x = self.dropout_act(self.activation(self.conv1(x)))
        #print("PrP1 size: ", x.shape)

        # For visualization
        first_layer_activations = x

        '''
        if not self.training:
            # global file_index
            if not os.path.exists("./activations/data-validation-" + str(file_index)):
                os.mkdir("./activations/data-validation-" + str(file_index))
            printActivations(x, orig, "data-validation-" + str(file_index))
            file_index = file_index + 1
        '''
        h, w = x.shape[2:]
        
        if self.BN and self.training:
            x = self.batchNormP1(x.view(self.batch_size, -1))
            x = x.view(self.batch_size, 32, h, w)
        elif self.BN:
            x = self.batchNormP1(x.view(1, -1))
            x = x.view(1, 32, h, w)
        
        # printActivations(x, orig)
        # print("\nx size: ", x.shape)
        # x = self.dropout1(x)
        
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        x = F.dropout(F.relu(self.convc1(x)), p=self.p2)
        # x = self.dropout_act(self.activation(self.convc1(x)))
        # x = self.dropout1a(x)
        
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        x = F.dropout(F.relu(self.convc2(x)), p=self.p2)
        # x = self.dropout_act(self.activation(self.convc2(x)))
        # x = self.dropout1b(x)

        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        x = F.dropout(F.relu(self.convc3(x)), p=self.p2)
        # x = self.dropout_act(self.activation(self.convc3(x)))

        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        x = F.dropout(F.relu(self.conv2(x)), p=self.p2)
        # x = self.dropout_act(self.activation(self.conv2(x)))

        
        if self.BN and self.training:
            x = self.batchNormP2(x.view(self.batch_size, -1))
            x = x.view(self.batch_size, 8, h, w)
        elif self.BN:
            x = self.batchNormP2(x.view(1, -1))
            x = x.view(1, 8, h, w)
        
        #print("PoP2 size: ", x.shape)
        # x = self.dropout2(x)
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.activation(self.conv3(x)))
        x = F.dropout(F.relu(self.conv3(x)), p=self.p2)

        # print("size: ", x.shape)

        
        if self.BN and self.training:
            x = self.batchNormP3(x.view(self.batch_size, -1))
            x = x.view(self.batch_size, 8, h, w)
        elif self.BN:
            x = self.batchNormP3(x.view(1, -1))
            x = x.view(1, 8, h, w)
        
        #  print("size: ", x.shape)

        # x = self.dropout3(x)
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.activation(self.conv4(x)))
        x = F.dropout(F.relu(self.conv4(x)),p=self.p2)
        # x = self.dropout4(x)


        if self.BN and self.training:
            x = self.batchNorm1(x.view(self.batch_size, -1))
            x = x.view(self.batch_size, -1)
        elif self.BN:
            x = self.batchNorm1(x.view(1, -1))
        
        x = x.view(self.batch_size, -1)
        
        '''
        elif self.training:
            print("size: ", x.shape)

            x = x.view(self.batch_size, -1)
        else:
            print("size: ", x.shape)

            x = x.view(self.batch_size, -1)
        '''
        # print("fart size: ", x.shape)
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.activation(self.linear1(x)))

        x = F.dropout(F.relu(self.linear1(x)), p=self.p2)

        # print("size: ", x.shape)
        
        if self.BN:
            x = self.batchNorm2(x)
        
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        x = F.dropout(F.relu(self.linear2(x)), p=self.p2)

        # x = self.dropout_act(self.activation(self.linear2(x)))
        
        if self.BN:
            x = self.batchNorm3(x)
        
        # print("size: ", x.shape)
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.activation(self.linear3(x)))
        x = F.dropout(F.relu(self.linear3(x)), p=self.p2)
        # print("size: ", x.shape)
        
        if self.BN:
            x = self.batchNorm4(x)
        
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.activation(self.linear4(x)))
        # print("size: ", x.shape)
        x = F.dropout(F.relu(self.linear4(x)), p=self.p2)

        if self.BN:
            x = self.batchNorm5(x)
        
        # x = self.dropout_inp(x)
        x = F.dropout(x, p=self.p1)
        # x = self.dropout_act(self.linear5(x))
        # print("size: ", x.shape)
        x = self.linear5(x)
        return x # torch.sum(x,dim=0)


