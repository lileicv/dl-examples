'''
使用 pytorch 实现简单的 gan 生成 mnist
    - 判别器结构简单点好，性能太强的话不容易收敛
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import os
from tqdm import tqdm
import cv2
import numpy as np

class Generator(nn.Module):
    def __init__(self, zdim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(zdim, 7*7*16)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1) # 14 14 64
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1) # 28 28 32
        self.conv3 = nn.Conv2d(16, 1, 3, 1, 1)  # 28 28 1
    
    def forward(self, z):
        x = F.relu(self.fc1(z)).view(-1, 16, 7, 7)          # 64  7  7
        x = F.relu(F.interpolate(self.conv1(x), [14,14]))   # 64 14 14
        x = F.relu(F.interpolate(self.conv2(x), [28,28]))   # 32 28 28
        x = torch.sigmoid(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 8, 5)
        self.fc1 = nn.Linear(4*4*8, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 12 12
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 8 8
        x = x.view(-1, 4*4*8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc2(x)
        #x = torch.sigmoid(self.fc2(x))
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
        Dloss = torch.pow(batchpx-1, 2).mean() + torch.pow(batchpg, 2).mean()
        Gloss = torch.pow(batchpg2-1, 2).mean()
        return Dloss, Gloss

if __name__=='__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    noise_dim = 10
    epochs = 100

    # 加载数据集
    trset = MNIST(root='/home/lilei/.pytorch/', train=True, transform=transforms.ToTensor())
    #teset = MNIST(root='/home/lilei/.pytorch/mnist', train=False, transform=transforms.ToTensor())
    trloader = DataLoader(dataset=trset, batch_size=128, shuffle=True)

    # 加载模型
    gan = GAN(noise_dim).cuda()

    Gopt = optim.Adam(gan.G.parameters(), lr=0.0003)
    Dopt = optim.Adam(gan.D.parameters(), lr=0.0003)

    # 开始训练
    for epoch in range(epochs):
        gan.train()
        Gloss_list, Dloss_list = [], []
        pbar = tqdm(trloader)
        for i,(batchx, _) in enumerate(pbar):
            batchx = batchx.cuda()
            Dloss, Gloss = gan(batchx)
            

            Dopt.zero_grad()
            Dloss.backward()
            Dopt.step()
            Gopt.zero_grad()
            Gloss.backward()
            Gopt.step()

            Gloss_list.append(Gloss.item())
            Dloss_list.append(Dloss.item())

            pbar.set_description('{}, loss(G/D) {:<.3f}/{:<.3f}'.format(epoch, np.mean(Gloss_list), np.mean(Dloss_list)))

        # 生成几张图片
        rows, cols = 8, 8
        batchz = torch.randn(rows*cols, noise_dim).cuda()
        batchg = gan.G(batchz).detach().cpu().numpy()
        bigimg = np.zeros([rows*28, cols*28])
        for i in range(rows):
            for j in range(cols):
                bigimg[i*28:i*28+28, j*28:j*28+28] = np.squeeze(batchg[i*rows+j])
        bigimg = (bigimg*255).astype('uint8')
        cv2.imwrite('./images/mnist-lsgan-pth-{}.jpg'.format(epoch), bigimg)




