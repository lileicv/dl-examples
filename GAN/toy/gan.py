'''
'''
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np
from tqdm import tqdm
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import argparse

from dataset import *
from models import *
from draw import generate_image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataname', type=str, default='25gaussians')
parser.add_argument('--noise-dim', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

name = 'gan'
dataname = args.dataname
noise_dim = args.noise_dim
epochs = args.epochs
iters_per_epoch = 100
batchsize = 512
Diter = 5 # 每 Diter 次，训练一次生成器 G

savefolder = './results/{}-{}'.format(dataname, name)
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

# Load Dataset
dataset = inf_train_gen(dataname, batchsize)

# Load Model
G = Generator(noise_dim).cuda()
D = Discriminator(sigmoid=True).cuda()
Gopt = optim.Adam(G.parameters(), lr=0.0001)
Dopt = optim.Adam(D.parameters(), lr=0.0001)

for epoch in range(epochs):
    Glosses = []
    Dlosses = []
    for i in range(iters_per_epoch):
        # 训练判别器
        for j in range(Diter):
            batchx = torch.tensor(next(dataset)).cuda()
            batchz = torch.randn(batchx.shape[0], noise_dim).cuda()
            batchg = G(batchz)
            batchpx = D(batchx)
            batchpg = D(batchg.detach())
            Dloss = F.binary_cross_entropy(batchpx, torch.ones_like(batchpx).cuda()) + \
                    F.binary_cross_entropy(batchpg, torch.zeros_like(batchpg).cuda())
            Dopt.zero_grad()
            Dloss.backward()
            Dopt.step()
            Dlosses.append(Dloss.item())
    
        # 训练生成器
        batchpg = D(batchg)
        Gloss = F.binary_cross_entropy(batchpg, torch.ones_like(batchpg).cuda())
        Gopt.zero_grad()
        Gloss.backward()
        Gopt.step()
        Glosses.append(Gloss.item())

    print('epoch {}, loss(D/G) {:<.3f}/{:<.3f}'.format(epoch, np.mean(Dlosses), np.mean(Glosses)))
    
    generate_image(next(dataset), G, D, savefolder + '/{}.png'.format(epoch))
