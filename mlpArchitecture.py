# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:34:09 2024

@author: jonat
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F

class shallowNN(nn.Module):
    def __init__(self):
        super(shallowNN, self).__init__()

        self.fc1 = nn.Linear(1024, 1000)  # First hidden layer with 1000 neurons (pp. 5)
        self.fc2 = nn.Linear(1000, 1000)  # Second hidden layer with 1000 neurons (pp. 5)
        self.fc3 = nn.Linear(1000, 1)     # Output layer with binary classification (pp.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = torch.sigmoid(self.fc3(x))  
        return x

