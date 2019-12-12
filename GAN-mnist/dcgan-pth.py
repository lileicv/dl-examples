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
        x = torch.sigmoid(self.fc2(x))
        return x

if __name__=='__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    noise_dim = 10
    epochs = 100

    # 加载数据集
    trset = MNIST(root='/home/lilei/.pytorch/', train=True, transform=transforms.ToTensor())
    #teset = MNIST(root='/home/lilei/.pytorch/mnist', train=False, transform=transforms.ToTensor())
    trloader = DataLoader(dataset=trset, batch_size=128, shuffle=True)

    # 加载模型
    G = Generator(noise_dim).cuda()
    D = Discriminator().cuda()

    Gopt = optim.Adam(G.parameters(), lr=0.0003)
    Dopt = optim.Adam(D.parameters(), lr=0.0003)

    # 开始训练
    for epoch in range(epochs):
        D.train()
        Gloss_list, Dloss_list = [], []
        pbar = tqdm(trloader)
        for i,(batchx, _) in enumerate(pbar):
            batchx = batchx.cuda()
            # 训练判别器
            batchz = torch.randn(batchx.shape[0], noise_dim).cuda()
            batchg = G(batchz)
            batchpx = D(batchx)
            batchpg = D(batchg)
            Dloss = F.binary_cross_entropy(batchpx, torch.ones_like(batchpx).cuda()) + \
                    F.binary_cross_entropy(batchpg, torch.zeros_like(batchpg).cuda())

            Dopt.zero_grad()
            Dloss.backward()
            Dopt.step()

            batchg = G(batchz)
            batchpg = D(batchg)
            Gloss = F.binary_cross_entropy(batchpg, torch.ones_like(batchpg).cuda())

            Gopt.zero_grad()
            Gloss.backward()
            Gopt.step()

            Gloss_list.append(Gloss.item())
            Dloss_list.append(Dloss.item())

            pbar.set_description('{}, loss(G/D) {:<.3f}/{:<.3f}'.format(epoch, np.mean(Gloss_list), np.mean(Dloss_list)))

        # 生成几张图片
        rows, cols = 8, 8
        batchz = torch.randn(rows*cols, noise_dim).cuda()
        batchg = G(batchz).detach().cpu().numpy()
        bigimg = np.zeros([rows*28, cols*28])
        for i in range(rows):
            for j in range(cols):
                bigimg[i*28:i*28+28, j*28:j*28+28] = np.squeeze(batchg[i*rows+j])
        bigimg = (bigimg*255).astype('uint8')
        cv2.imwrite('./images/mnist-dcgan-pth-{}.jpg'.format(epoch), bigimg)




