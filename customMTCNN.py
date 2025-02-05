import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Tuple, List
from customMTCNN import CustomMTCNN

class PNet(nn.module):
    """Proposal Network (P-Net) - First stage of MTCNN"""
    def __init__(self):
        super(PNet, self).__init__()

        #Feature Extraction Layer
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(32)

        #Detection Branch
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.softmax4_1 = nn.Softmax(dim=1)

        # Bounding box regression branch
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        
        # Landmark localization branch
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        # Face classification
        det = self.conv4_1(x)
        det = self.softmax4_1(det)
        
        # Bounding box regression
        box = self.conv4_2(x)
        
        # Facial landmark localization
        landmark = self.conv4_3(x)
        
        return det, box, landmark

class RNet(nn.module):
    """Refinement Network (R-Net) - Second stage of MTCNN"""
    def __init__(self):
        super(RNet, self).__init__()