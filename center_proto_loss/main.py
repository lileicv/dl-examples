'''
基于 pytorch 在 mnist 数据集上测试 center loss
'''
import numpy as np
from tqdm import tqdm
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # ==> 28,28,32
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)    # ==> 28,28,32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    # ==> 14,14,64
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)    # ==> 14,14,64
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)   # ==> 7,7,128
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  # ==> 7,7,128
        self.fc1 = nn.Linear(3*3*128, 2)
        self.fc2 = nn.Linear(2, 10, False)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(self.conv5(x))
        x = F.relu(F.max_pool2d(self.conv6(x), 2))

        x = x.view(-1, 3*3*128)
        f = self.fc1(x)
        y = F.log_softmax(self.fc2(f), dim=1)
        return y, f

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.embedding = nn.Embedding(10, 2)
    
    def forward(self, feature, label):
        center = self.embedding(label)
        center_loss = torch.pow(feature-center, 2).sum(1).mean()
        return center_loss

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.embedding = nn.Embedding(10, 2)
        # self.centers = torch.nn.Parameter(torch.randn(10, 2)*1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        centers = torch.unsqueeze(self.embedding.weight, 0)
        x = -1*torch.pow(x-centers, 2).sum(-1)
        y = F.log_softmax(x, 1)
        return y, x

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='center_loss')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # 全局参数
    epochs = args.epochs
    method = args.method # 'center_loss' or 'proto_loss'
    assert(method in ['center_loss', 'proto_loss'])
    print('method: {}'.format(method))

    # 加载数据
    trset = datasets.MNIST(root='/home/lilei/.pytorch', train=True, transform=transforms.ToTensor())
    teset = datasets.MNIST(root='/home/lilei/.pytorch', train=False, transform=transforms.ToTensor())
    trloader = DataLoader(trset, batch_size=128, shuffle=True, num_workers=8)
    teloader = DataLoader(teset, batch_size=128, num_workers=8)

    # 加载模型
    main_net = MainNet().cuda()
    cent_net = CenterLoss().cuda()
    proto_net = ProtoNet().cuda()
    optim1 = optim.Adam(main_net.parameters(), lr=0.001)
    optim2 = optim.Adam(cent_net.parameters(), lr=0.001)
    optim3 = optim.Adam(proto_net.parameters(), lr=0.001)

    for epoch in range(epochs):
        main_net.train()
        cent_net.train()

        # 训练
        xent_losses = []
        cent_losses = []
        acces = []
        pbar = tqdm(trloader)
        for i, (batchx, batchy) in enumerate(pbar):
            batchx, batchy = batchx.cuda(), batchy.cuda()
            logit, batchf = main_net(batchx)
            if method=='center_loss':
                xent_loss = F.nll_loss(logit, batchy)
                cent_loss = cent_net(batchf, batchy)
                loss = xent_loss + 0.2*cent_loss
            elif method=='proto_loss':
                logit, bbbb = proto_net(batchf)
                # if epoch==5:
                #     print(bbbb)
                #     print(logit)
                #     1/0
                xent_loss = F.nll_loss(logit, batchy)
                cent_loss = torch.tensor(0)
                loss = xent_loss
            
            optim1.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()
            loss.backward()
            optim1.step()
            optim2.step()
            optim3.step()
            accuracy = torch.eq(logit.argmax(1), batchy).float().mean().item()

            xent_losses.append(xent_loss.item())
            cent_losses.append(cent_loss.item())
            acces.append(accuracy)
            
            pbar.set_description('epoch:{}, loss(x/c):{:<.4f}/{:<.4f}, acc={:<.4f}'.format(epoch, np.mean(xent_losses), np.mean(cent_losses), np.mean(acces)))
        print(bbbb[0:4])
        print(logit[0:4])

        # 预测
        main_net.eval()
        ete = []
        yte = []
        for batchx, batchy in teloader:
            _, batchf = main_net(batchx.cuda())
            ete.append(batchf.cpu().detach().numpy())
            yte.append(batchy.cpu().numpy())
        ete = np.concatenate(ete, 0)
        yte = np.concatenate(yte, 0)

        # 画图
        if method=='center_loss':
            centers = cent_net.embedding.weight.cpu().detach().numpy()
        elif method=='proto_loss':
            centers = proto_net.embedding.weight.cpu().detach().numpy()

        for j in range(10):
            x = ete[yte==j, 0]
            y = ete[yte==j, 1]
            plt.scatter(x, y, label='{}'.format(j), marker='.', alpha=0.1)
            plt.scatter(centers[j,0], centers[j,1], marker='.', c='b')
        plt.savefig('./figures/{}_{}.png'.format(method, epoch))
        plt.close()
