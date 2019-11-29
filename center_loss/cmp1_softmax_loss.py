'''
- 使用普通 softmax loss 计算 mnist 分类
- 可视化 mnist 样本在特征空间的分布
'''

import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LearningRateScheduler

embedding = Sequential([
    L.Conv2D(32, 3, 1, 'same', activation='relu', input_shape=[28,28,1]),
    L.MaxPooling2D(),
    L.Conv2D(64, 3, 1, 'same', activation='relu'),
    L.MaxPooling2D(),
    L.Conv2D(128, 3, 1, 'same', activation='relu'),
    L.MaxPooling2D(),
    L.Flatten(),
    L.Dropout(0.5),
    L.Dense(2)
])

net = Sequential([
    embedding,
    L.Dense(10, activation='softmax')
])

(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = np.expand_dims(xtr, -1).astype('float32')/255.
xte = np.expand_dims(xte, -1).astype('float32')/255.

net.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
net.fit(xtr, ytr, epochs=3, batch_size=128, validation_data=[xte, yte])
ete = embedding.predict(xte)

for j in range(10):
    x = ete[yte==j, 0]
    y = ete[yte==j, 1]
    plt.scatter(x, y, label='{}'.format(j))
plt.savefig('./figures/softmax_loss.png')
plt.close()
