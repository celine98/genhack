from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.utils as vutils


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device: {}".format(device))


class Disciminator(nn.Module):

    def __init__(self, input_size=4):
        super(Disciminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.nl1 = nn.LeakyReLU(0.25)
        self.fc2 = nn.Linear(32, 16)
        self.nl2 = nn.LeakyReLU(0.125)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        o = self.nl1(self.fc1(x))
        o = self.nl2(self.fc2(o))
        return torch.sigmoid(self.fc3(o))


class Generator(nn.Module):

    def __init__(self, z_size=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_size, 128)
        self.nl1 = nn.LeakyReLU(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.nl2 = nn.LeakyReLU(0.125)
        self.fc3 = nn.Linear(64, 4)
    
    def forward(self, x):
        o = self.nl1(self.fc1(x))
        o = self.nl2(self.fc2(o))
        return self.fc3(o)
        

if __name__ == "__main__":
    
    G = Generator().to(device)
    G.load_state_dict(torch.load("model.pt"))
    G.to(device)
    df = pd.read_csv("noise.csv")
    noise = torch.tensor(df.values).to(device).float()
    outputs = G(noise).cpu().detach().numpy()
    pd.DataFrame(outputs).to_csv("outputs.csv", index = None)
