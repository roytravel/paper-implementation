# -*- coding:utf-8 -*-
# reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

import os
import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.init_size = option.image_size // 4
        self.l1 = nn.Sequential(nn.Linear(option.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, option.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        image = self.conv_blocks(out)
        return image


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        def discriminator_block(input_filters, output_filters, bn=True):
            block = [nn.Conv2d(input_filters, output_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(output_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(option.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128))

        ds_size = option.image_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())


    def forward(self, image):
        out = self.model(image)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class Preprocessing(object):
    def __init__(self):
        os.makedirs("exclude/data/mnist", exist_ok=True)
    
    def weight_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == "__main__":
    # Add option
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.3, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--image_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    option = parser.parse_args()
    print(option)

    G = Generator()
    D = Discriminator()
    P = Preprocessing()

    cuda = True if torch.cuda.is_available() else False
    adversarial_loss = torch.nn.BCELoss()

    if cuda:
        G.cuda()
        D.cuda()
        adversarial_loss.cuda()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=option.lr, betas=(option.b1, option.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=option.lr, betas=(option.b1, option.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    G.apply(P.weight_init_normal)
    D.apply(P.weight_init_normal)


    dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "exclude/data/mnist",
                train=True, 
                download=True, 
                transform=transforms.Compose([transforms.Resize(option.image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            ),
        batch_size = option.batch_size,
        shuffle = True)


    for epoch in range(option.n_epochs):

        for i, (images, _) in enumerate(dataloader):
            valid = Variable(Tensor(images.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(images.shape[0], 1).fill_(0.0), requires_grad=False)

            real_images = Variable(images.type(Tensor))

            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], option.latent_dim))))

            gen_images = G(z)
            generator_loss = adversarial_loss(D(gen_images), valid)
            generator_loss.backward()
            optimizer_G.step()


            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_images), valid)
            fake_loss = adversarial_loss(D(gen_images.detach()), fake)
            discrimator_loss = (real_loss + fake_loss) / 2
            discrimator_loss.backward()
            optimizer_D.step()

            print (f"[Epoch {epoch}/{option.n_epochs}] [Batch {i+1}/{len(dataloader)}] [D loss: {discrimator_loss.item()}] [G loss: {generator_loss.item()}]")

            batchs_done = epoch * len(dataloader) + i
            if batchs_done % option.sample_interval == 0:
                save_image(gen_images.data[:25], "exclude/dcgan/%d.png" % batchs_done, nrow=5, normalize=True)


