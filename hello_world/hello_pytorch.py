'''
pytorch 入门
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np
from tensorflow.keras.datasets import mnist

class MnistDataset(Dataset):
    def __init__(self, phase):
        ''' Load dataset
            phase ==> ['train', 'test']
        '''
        (xtr,ytr),(xte,yte) = mnist.load_data()
        if phase == 'train':
            self.data = np.expand_dims(xtr.astype('float32')/255., axis=1)
            self.label = ytr.astype('int64')
        elif phase == 'test':
            self.data = np.expand_dims(xte.astype('float32')/255., axis=1)
            self.label = yte.astype('int64')
    def __len__(self,):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.drop2d = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*32, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 12 x 12
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 8 x 8
        x = self.drop2d(x)                          # 4 x 4
        x = x.view(-1, 4*4*32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x


# Build Dataset
tr_mnist = MnistDataset('train')
te_mnist = MnistDataset('test')

tr_loader = DataLoader(tr_mnist, batch_size=32, shuffle=True, num_workers=5)
te_loader = DataLoader(te_mnist, batch_size=32, shuffle=False, num_workers=5)

# Build Model
model = Net()
model.cuda()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    # Train
    model.train()
    loss_list = []
    for i, (batchx, batchy) in enumerate(tr_loader):
        batchx, batchy = batchx.cuda(), batchy.cuda()
        batchx, batchy = Variable(batchx), Variable(batchy)
        
        optimizer.zero_grad()
        logit = model(batchx)
        loss = F.nll_loss(logit, batchy)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if i%100 == 0:
            print('\rEpoch: {}, iter: {}, loss: {:<.4f}'.format(epoch, i, np.mean(loss_list)), end='')

    correct = 0
    model.eval()
    for i, (batchx, batchy) in enumerate(te_loader):
        batchx, batchy = batchx.cuda(), batchy.cuda()
        batchx, batchy = Variable(batchx), Variable(batchy)

        logit = model(batchx)
        logit = logit.max(1)[1]
        correct += logit.eq(batchy).sum().item()
   
    acc = 1.0 * correct / len(te_loader.dataset)
    print(', test accuracy: {}'.format(acc))

