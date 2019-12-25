'''
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, idim=2, hdim=512, odim=2):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(idim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, odim)
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, idim=2, hdim=512, odim=1, sigmoid=False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(idim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, odim)
        )
        self.sigmoid = sigmoid
    
    def forward(self, x):
        x = self.main(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

# class GAN(nn.Module):
#     def __init__(self, noise_dim):
#         super(GAN, self).__init__()
#         self.G = Generator(noise_dim)
#         self.D = Discriminator()
#         self.noise_dim = noise_dim
#     def forward(self, batchx):
#         batchz = torch.randn(batchx.shape[0], self.noise_dim).cuda()
#         batchg = self.G(batchz)
#         batchpx = self.D(batchx)
#         batchpg = self.D(batchg.detach())
#         batchpg2 = self.D(batchg)
#         Dloss = F.binary_cross_entropy(batchpx, torch.ones_like(batchpx).cuda()) + \
#                 F.binary_cross_entropy(batchpg, torch.zeros_like(batchpg).cuda())
#         Gloss = F.binary_cross_entropy(batchpg2, torch.ones_like(batchpg).cuda())
#         return Dloss, Gloss

# class LSGAN(nn.Module):
#     def __init__(self, noise_dim):
#         super(LSGAN, self).__init__()
#         self.G = Generator(noise_dim)
#         self.D = Discriminator()
#         self.noise_dim = noise_dim
#     def forward(self, batchx):
#         batchz = torch.randn(batchx.shape[0], self.noise_dim).cuda()
#         batchg = self.G(batchz)
#         batchpx = self.D(batchx, sigmoid=False)
#         batchpg = self.D(batchg.detach(), sigmoid=False)
#         batchpg2 = self.D(batchg, sigmoid=False)
#         Dloss = torch.pow(batchpx-1, 2).mean() + torch.pow(batchpg, 2).mean()
#         Gloss = torch.pow(batchpg2-1, 2).mean()
#         return Dloss, Gloss

# class WGAN_GP(nn.Module):
#     def __init__(self, noise_dim):
#         super(WGAN_GP, self).__init__()
#         self.G = Generator(noise_dim)
#         self.D = Discriminator()
#         self.noise_dim = noise_dim

#     def calc_gradient_penalty(self, netD, real_data, fake_data):
#         alpha = torch.rand(real_data.shape[0], 1)
#         alpha = alpha.expand(real_data.size())
#         alpha = alpha.cuda()

#         interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#         interpolates = interpolates.cuda()
#         interpolates = autograd.Variable(interpolates, requires_grad=True)

#         disc_interpolates = netD(interpolates)

#         gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                                 grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
#                                 create_graph=True, retain_graph=True, only_inputs=True)[0]

#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 0.1
#         return gradient_penalty

#     def forward(self, batchx):
#         batchz = torch.randn(batchx.shape[0], self.noise_dim).cuda()
#         batchg = self.G(batchz)
#         batchpx = self.D(batchx, sigmoid=False)
#         batchpg = self.D(batchg.detach(), sigmoid=False)
#         batchpg2 = self.D(batchg, sigmoid=False)
#         Dloss = -batchpx.mean() + batchpg.mean() + self.calc_gradient_penalty(self.D, batchx, batchg)
#         Gloss = -batchpg2.mean()
#         return Dloss, Gloss
