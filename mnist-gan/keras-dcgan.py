#!/usr/bin/env python
# coding: utf-8

# 使用 mnist 数据集测试 gan 算法

# In[1]:


import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from tensorflow.keras.datasets import mnist
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class DCGAN():
    def __init__(self):
        D = Sequential([
            Conv2D(32, 3, padding='same', activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
            ])
        G = Sequential([
            Dense(32*7*7, input_shape=(10,), activation='relu'),
            Reshape((7,7,32)),
            UpSampling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            UpSampling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(1, 3, padding='same')
            ])
        D.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        z = Input(shape=(10,))
        img = G(z)
        D.trainable = False
        valid = D(img)
        combine = Model(z,valid)
        combine.compile(loss='binary_crossentropy', optimizer='adam')
        
        self.G = G
        self.D = D
        self.combine = combine
    def train(self, nepoch, batchsize=128, save_pt=[0, 50, 100]):
        (xtr,_),(_,_) = mnist.load_data()
        xtr = np.expand_dims(xtr, 3)/255.
        valid = np.ones((batchsize, 1))
        fake = np.zeros((batchsize, 1))
        d_loss_epochs = []
        g_loss_epochs = []
        for epoch in range(nepoch):
            perm = np.random.permutation(xtr.shape[0])
            d_loss_list = []
            g_loss_list = []
            for idx in range(0,perm.shape[0],batchsize):
                imgs = xtr[idx:idx+batchsize]
                if imgs.shape[0] == valid.shape[0]:
                    noise = np.random.normal(0,1,(batchsize, 10))
                    # 训练判别器
                    gen_imgs = self.G.predict(noise)
                    loss_real = self.D.train_on_batch(imgs, valid)[0]
                    loss_fake = self.D.train_on_batch(gen_imgs, fake)[0]
                    d_loss = (loss_real+loss_fake)/2.
                    # 训练生成器
                    g_loss = self.combine.train_on_batch(noise, valid)
                    d_loss_list.append(d_loss)
                    g_loss_list.append(g_loss)
            d_loss = np.mean(d_loss_list)
            g_loss = np.mean(g_loss_list)
            print('epoch:{}, D loss:{:.6f}, G:loss:{:.6f}'.format(epoch, d_loss, g_loss))
            d_loss_epochs.append(d_loss)
            g_loss_epochs.append(g_loss)
        return d_loss_epochs, g_loss_epochs

    def gen_imgs(self, r=6, c=12):
        r, c = 6, 12
        noise = np.random.normal(0, 1, (r * c, 10))
        gen_imgs = self.G.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        #plt.show()
        plt.savefig('./images/keras-dcgan.png')
        plt.close()


# In[3]:


dcgan = DCGAN()
d_loss, g_loss = dcgan.train(30)


# In[6]:


plt.figure(figsize=[12,3])
plt.subplot(1,2,1, title='D loss')
plt.plot(d_loss)
plt.subplot(1,2,2, title='G loss')
plt.plot(g_loss)
plt.savefig('./train_loss.png')


# In[8]:


dcgan.gen_imgs()

