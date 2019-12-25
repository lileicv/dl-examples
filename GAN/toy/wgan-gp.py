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

name = 'wgan-gp'
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
D = Discriminator().cuda()
Gopt = optim.Adam(G.parameters(), lr=0.0001)
Dopt = optim.Adam(D.parameters(), lr=0.0001)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 0.1
    return gradient_penalty

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
            Dloss = -batchpx.mean() + batchpg.mean() + calc_gradient_penalty(D, batchx, batchg)
            Dopt.zero_grad()
            Dloss.backward()
            Dopt.step()
            Dlosses.append(Dloss.item())
    
        # 训练生成器
        batchpg = D(batchg)
        Gloss = -batchpg.mean()
        Gopt.zero_grad()
        Gloss.backward()
        Gopt.step()
        Glosses.append(Gloss.item())

    print('epoch {}, loss(D/G) {:<.3f}/{:<.3f}'.format(epoch, np.mean(Dlosses), np.mean(Glosses)))
    
    generate_image(next(dataset), G, D, savefolder + '/{}.png'.format(epoch))
