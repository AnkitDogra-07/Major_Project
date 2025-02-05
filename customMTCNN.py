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