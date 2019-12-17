'''
基于梯度反转层实现 DCGAN
梯度反转层效果不如原始的GAN，不知道为什么？

综合来看，在对抗类的任务中，不建议使用梯度反转层，这个实现虽然方便，但是调试不方便，在GAN中，有两个loss，通过这两个loss，可以看出当前训练是否正常，这个只有一个loss，很难从这一个loss中看出是否正常

梯度反转层很难生成正常的mnist样本

结论：这个模型比GAN更难训练，以后有需求还是找GAN更好
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

class ReverseLayerF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()*ctx.alpha
        return output, None

class DCGAN(nn.Module):
    def __init__(self, zdim):
        super(DCGAN, self).__init__()
        # 生成器
        self.fc1 = nn.Linear(zdim, 7*7*16)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1) # 14 14 64
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1) # 28 28 32
        self.conv3 = nn.Conv2d(16, 1, 3, 1, 1)  # 28 28 1
        # 判别器
        self.conv4 = nn.Conv2d(1, 8, 5)
        self.conv5 = nn.Conv2d(8, 8, 5)
        self.fc2 = nn.Linear(4*4*8, 10)
        self.fc3 = nn.Linear(10, 1)
        # 梯度反转层
        self.reverse = ReverseLayerF.apply

    def G(self, z):
        x = F.relu(self.fc1(z)).view(-1, 16, 7, 7)          # 64  7  7
        x = F.relu(F.interpolate(self.conv1(x), [14,14]))   # 64 14 14
        x = F.relu(F.interpolate(self.conv2(x), [28,28]))   # 32 28 28
        x = torch.sigmoid(self.conv3(x))
        return x
    
    def D(self, x):
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # 12 12
        x = F.relu(F.max_pool2d(self.conv5(x), 2))  # 8 8
        x = x.view(-1, 4*4*8)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, self.training)
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def forward(self, batchx, alpha):
        batchz = torch.randn(batchx.shape[0], noise_dim).cuda()
        batchg = self.G(batchz)
        batchg = self.reverse(batchg, alpha)
        # batchg = ReverseLayerF.apply(batchg, 1)
        batchpx = self.D(batchx)
        batchpg = self.D(batchg)
        loss = F.binary_cross_entropy(batchpx, torch.ones_like(batchpx).cuda()) + \
                F.binary_cross_entropy(batchpg, torch.zeros_like(batchpg).cuda())
        return loss        


if __name__=='__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    noise_dim = 10
    epochs = 100

    # 加载数据集
    trset = MNIST(root='/home/lilei/.pytorch/', train=True, transform=transforms.ToTensor(), download=True)
    #teset = MNIST(root='/home/lilei/.pytorch/mnist', train=False, transform=transforms.ToTensor())
    trloader = DataLoader(dataset=trset, batch_size=128, shuffle=True)

    # 加载模型
    gan = DCGAN(noise_dim).cuda()

    opt = optim.Adam(gan.parameters(), lr=0.003)

    # 开始训练
    for epoch in range(epochs):
        alpha = epoch * 0.1
        gan.train()
        loss_list = []
        pbar = tqdm(trloader)
        for i,(batchx, _) in enumerate(pbar):
            batchx = batchx.cuda()
            loss = gan(batchx, 0.5)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_list.append(loss.item())

            pbar.set_description('{}, loss {:<.3f}'.format(epoch, np.mean(loss_list)))

        # 生成几张图片
        rows, cols = 8, 8
        batchz = torch.randn(rows*cols, noise_dim).cuda()
        batchg = gan.G(batchz).detach().cpu().numpy()
        bigimg = np.zeros([rows*28, cols*28])
        for i in range(rows):
            for j in range(cols):
                bigimg[i*28:i*28+28, j*28:j*28+28] = np.squeeze(batchg[i*rows+j])
        bigimg = (bigimg*255).astype('uint8')
        cv2.imwrite('./images/{}.jpg'.format(epoch), bigimg)




