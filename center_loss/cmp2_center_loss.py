'''
- 使用普通 softmax loss 计算 mnist 分类
- 可视化 mnist 样本在特征空间的分布
'''
import os
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler

class MainNet(tf.keras.Model):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = L.Conv2D(32, 3, 1, 'same', activation='relu')
        self.conv2 = L.Conv2D(32, 3, 1, 'same', activation='relu')
        self.conv3 = L.Conv2D(64, 3, 1, 'same', activation='relu')
        self.conv4 = L.Conv2D(64, 3, 1, 'same', activation='relu')
        self.conv5 = L.Conv2D(128, 3, 1, 'same', activation='relu')
        self.conv6 = L.Conv2D(128, 3, 1, 'same', activation='relu')
        self.fc1 = L.Dense(2)
        self.fc2 = L.Dense(10, activation='softmax')
    
    def call(self, x, is_training=False):
        h = self.conv1(x)
        h = self.conv2(h)
        h = L.MaxPooling2D()(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = L.MaxPooling2D()(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = L.MaxPooling2D()(h)
        h = L.Flatten()(h)
        h = L.Dropout(0.5)(h, training=is_training)
        f = self.fc1(h)
        y = self.fc2(f)
        return y, f

class CenterLoss(tf.keras.Model):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.embedding = L.Embedding(10, 2)

    def call(self, feature, label):
        center = self.embedding(label)
        cent_loss = tf.reduce_mean(tf.reduce_sum(tf.square((feature - center)), 1))
        return cent_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = np.expand_dims(xtr, -1).astype('float32')/255.
xte = np.expand_dims(xte, -1).astype('float32')/255.
trset = tf.data.Dataset.from_tensor_slices((xtr, ytr)).batch(128).shuffle(100)
teset = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(128)

main_net = MainNet()
center_loss = CenterLoss()

lr_reduce = lambda e: 0.005 if e<10 else 0.001
optim1 = Adam(0.01)
optim2 = Adam(0.1)
for epoch in range(50):
    lr = lr_reduce(epoch)
    optim1 = Adam(lr)
    cent_losses = []
    xent_losses = []
    acces = []
    loss_weight=0.2
    for i,(batchx, batchy) in enumerate(trset):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            batchp, batchf = main_net(batchx, is_training=True)
            cent_loss = center_loss(batchf, batchy)
            xent_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batchy, batchp))
            loss = xent_loss + loss_weight * cent_loss
            accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batchy, batchp))

            xent_losses.append(xent_loss.numpy())
            cent_losses.append(cent_loss.numpy())
            acces.append(accuracy.numpy()*100)

        grads = tape1.gradient(loss, main_net.variables)
        optim1.apply_gradients(zip(grads, main_net.variables))
        grads = tape2.gradient(loss, center_loss.variables)
        optim2.apply_gradients(zip(grads, center_loss.variables))
        print('\r epoch: {}, lr: {},  loss(xent/center): {:<.4f}/{:<.4f}, acc: {:<2.2f}({:<2.2f})%'.format(epoch, lr, np.mean(xent_losses), np.mean(cent_losses), acces[-1], np.mean(acces)), end=' ')
    print()
    
    if 1:
        _, ete = main_net.predict(xte)

        for j in range(10):
            x = ete[yte==j, 0]
            y = ete[yte==j, 1]
            plt.scatter(x, y, label='{}'.format(j))
        plt.savefig('./figures/center_loss_{}.png'.format(epoch))
        plt.close()
