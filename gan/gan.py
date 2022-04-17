# -*- coding:utf-8 -*-
# reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
import os
import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(input_feature, output_feature, normalize=True):
            """
            role: create layers
            structure: Linear -> BatchNorm -> LeakyReLU
            """
            layers = [nn.Linear(input_feature, output_feature)]

            if normalize:
                layers.append(nn.BatchNorm1d(output_feature, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(option.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh())

    
    def forward(self, z):
        image = self.model(z)
        image = image.view(image.size(0), *image_shape)
        return image



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),)

    def forward(self, image):
        image_flat = image.view(image.size(0), -1)
        validity = self.model(image_flat)
        return validity


if __name__ == "__main__":

    # Add option
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.3, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--image_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    option = parser.parse_args()
    print(option)

    # Define image shape
    image_shape = (option.channels, option.image_size, option.image_size)

    # Check if cuda is available
    cuda = True if torch.cuda.is_available() else False

    # Define adversarial loss for Generator G.
    adversarial_loss = torch.nn.BCELoss()

    # Create Instance
    G = Generator()
    D = Discriminator()


    if cuda:
        G.cuda()
        D.cuda()
        adversarial_loss.cuda()

    # Create directories if not exist
    os.makedirs('exclude/data/mnist', exist_ok=True)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(datasets.MNIST(
        "exclude/data/mnist",
        train=True,
        download=True,
        transform = transforms.Compose([transforms.Resize(option.image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])),
        batch_size = option.batch_size,
        shuffle=True)


    # Define optimizer for Generator and Discriminator
    optionimizer_G = torch.optim.Adam(G.parameters(), lr=option.lr, betas=(option.b1, option.b2))
    optionimizer_D = torch.optim.Adam(D.parameters(), lr=option.lr, betas=(option.b1, option.b2))

    # Create Tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FlaotTensor

    for epoch in range(option.n_epochs):
        for i, (images, _) in enumerate(dataloader):
            # Create tensor that size is images.size(0)
            """
            Form as below if valid.
            tensor([[1.],
                    [1.],
                    ...
                    [1.]], device='cuda:0')
            """
            valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)
            real_images = Variable(images.type(Tensor))
            
            """
            To avoid tracking from autograd.
            It directly manipulate Tensor to 0.
            because gradient is accumulated in the buffer every time .backward() is called.
            """
            optionimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], option.latent_dim))))

            # Generate a batch of images. Form: (batch_size, 1, 28, 28)
            generated_images = G(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(D(generated_images), valid)
            g_loss.backward()
            optionimizer_G.step()


            """ Train Discriminator """
            optionimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(D(real_images), valid)
            fake_loss = adversarial_loss(D(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optionimizer_D.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, option.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % option.sample_interval == 0:
                pass
                #save_image(generated_images.data[:25], "exclude/images/Test%d.png" % batches_done, nrow=5, normalize=True)
