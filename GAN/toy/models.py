'''
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, zdim=1):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(zdim, 10)
        self.fc2 = nn.Linear(10, 2)
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x, sigmoid=True):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x

class GAN(nn.Module):
    def __init__(self, noise_dim):
        super(GAN, self).__init__()
        self.G = Generator(noise_dim)
        self.D = Discriminator()
    def forward(self, x):
        batchz = torch.randn(batchx.shape[0], noise_dim).cuda()
        batchg = self.G(batchz)
        batchpx = self.D(batchx)
        batchpg = self.D(batchg.detach())
        batchpg2 = self.D(batchg)
        Dloss = F.binary_cross_entropy(batchpx, torch.ones_like(batchpx).cuda()) + \
                F.binary_cross_entropy(batchpg, torch.zeros_like(batchpg).cuda())
        Gloss = F.binary_cross_entropy(batchpg2, torch.ones_like(batchpg).cuda())
        return Dloss, Gloss

