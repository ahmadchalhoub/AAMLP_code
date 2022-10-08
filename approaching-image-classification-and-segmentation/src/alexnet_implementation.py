'''
This script demonstrates the implementation of AlexNet in PyTorch.
PyTorch uses the following notations:
           BS: Batch Size
           C: Channel
           H: Height
           W: Width
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # first layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # second layer
        self.conv2 = nn.conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # third layer
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # fourth layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
        kernel_size=3, stride=1, padding=1)

        # fifth layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
        kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # three dense layer (layers 6, 7, and 8)
        self.fc1 = nn.Linear(
            in_features=9216,
            out_features=4096
        )
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            in_features=4096,
            out_features=4096
        )
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(
            in_features=4096,
            out_features=1000
        )
