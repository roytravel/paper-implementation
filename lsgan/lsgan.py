# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from torchvision import datasets
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.model = self._build_layers()

    def _build_layers(self):
        pass

    def forward(self):
        pass

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.model = self._build_layers()

    def _build_layers():
        pass

    def forward(self):
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--some", dest="some",help="some")
    option = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    os.makedirs("../exclude/lsgan/", exist_ok=True)

    G = Generator()
    D = Discriminator()

    print (cuda)

    # Optimizer 
    optimizer_G = torch.optim.Adam()
    optimizer_D = torch.optim.Adam()

    # Loss
    adversarial_loss = nn.BCELoss()

    if cuda:
        G.cuda()
        D.cuda()
        adversarial_loss.cuda()

    dataloader = DataLoader(dataset="MNIST", batch_size=option.batch_size, shuffle=True)

    for epoch in tqdm(len(dataloader)):
        pass


