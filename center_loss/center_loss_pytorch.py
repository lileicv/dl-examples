'''
基于 pytorch 在 mnist 数据集上测试 center loss
'''
import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
        self.fc2 = nn.Linear(2, 10)
    
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

if __name__=='__main__':

    # 全局参数
    epochs = 10

    # 加载数据
    (xtr, ytr), (xte, yte) = mnist.load_data()
    xtr = np.expand_dims(xtr, 1).astype('float32')/255.
    xte = np.expand_dims(xte, 1).astype('float32')/255.
    trset = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr).long())
    teset = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte).long())
    trloader = DataLoader(trset, batch_size=128, shuffle=True, num_workers=8)
    teloader = DataLoader(teset, batch_size=128, num_workers=8)

    # 加载模型
    main_net = MainNet().cuda()
    cent_net = CenterLoss().cuda()
    optim1 = optim.Adam(main_net.parameters(), lr=0.001)
    optim2 = optim.Adam(cent_net.parameters(), lr=0.01)

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
            xent_loss = F.nll_loss(logit, batchy)
            cent_loss = cent_net(batchf, batchy)
            loss = xent_loss + 0.2*cent_loss
            optim1.zero_grad()
            optim2.zero_grad()
            loss.backward()
            optim1.step()
            optim2.step()
            accuracy = torch.eq(logit.argmax(1), batchy).float().mean().item()

            xent_losses.append(xent_loss.item())
            cent_losses.append(cent_loss.item())
            acces.append(accuracy)
            pbar.set_description('epoch:{}, loss(x/c):{:<.4f}/{:<.4f}, acc={:<.4f}'.format(epoch, np.mean(xent_losses), np.mean(cent_losses), np.mean(acces)))
    
        # 预测
        main_net.eval()
        ete = []
        for batchx, batchy in teloader:
            _, batchf = main_net(batchx.cuda())
            ete.append(batchf.cpu().detach().numpy())
        ete = np.concatenate(ete, 0)

        # 画图
        for j in range(10):
            x = ete[yte==j, 0]
            y = ete[yte==j, 1]
            plt.scatter(x, y, label='{}'.format(j), marker='.', alpha=0.1)
        plt.savefig('./figures/center_loss_{}.png'.format(epoch))
        plt.close()