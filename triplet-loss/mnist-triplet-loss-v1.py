'''
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def build_triplet(xtr, ytr, nb=0):
    ''' 构建三元组数据集[x1,x2,y1]
        其中，x1 x2 属于同一类，y1属于其他类别
        如果 nb == 0，那么三元组的个数等于 len(xtr)
        否则从三元组中 sample 出 nb 个
    '''
    x1 = xtr
    y1 = xtr.copy()
    np.random.shuffle(y1)
    x2idx = np.zeros(len(xtr), dtype='int32')
    for j in set(ytr):
        j_idx = np.where(ytr==j)[0]
        j_idx_rand = j_idx.copy()
        np.random.shuffle(j_idx_rand)
        x2idx[j_idx] = j_idx_rand
    x2 = xtr[x2idx]
    perm = np.random.permutation(len(x1))
    if nb>0:
        perm = perm[0:nb]
    return x1[perm], x2[perm], y1[perm]

def dist(x1,x2):
    # return tf.sqrt(tf.reduce_mean(tf.square(x1-x2), 1)) # 这里如果加上 tf.sqrt 的话，模型不收敛，不清楚为什么
    return tf.reduce_mean(tf.square(x1-x2), 1)

def vis_feature(feats, labels, vis_path):
    perm = np.random.permutation(len(feats))[0:1000]
    feats = feats[perm]
    labels = labels[perm]

    feats = TSNE(n_components=2).fit_transform(feats)
    # pca = PCA(n_components=2)
    # feats = pca.fit_transform(feats)
    plt.figure()
    for l in set(labels):
        feats_l = feats[labels==l]
        plt.scatter(feats_l[:,0], feats_l[:,1], label='{}'.format(l))
    plt.savefig(vis_path)
    plt.close()

(xtr,ytr),(xte,yte) = mnist.load_data()
xtr = np.expand_dims(xtr, -1).astype('float32')/255.
xte = np.expand_dims(xte, -1).astype('float32')/255.

x1,x2,y1 = build_triplet(xtr,ytr)
triplet = tf.data.Dataset.from_tensor_slices((x1,x2,y1)).batch(64).shuffle(100)

E = Sequential([
    L.Conv2D(32, 3, 1, 'same', activation='relu', input_shape=(28,28,1)),
    L.MaxPooling2D(),
    L.Conv2D(32, 3, 1, 'same', activation='relu'),
    L.MaxPooling2D(),
    L.Flatten(),
    L.Dense(128, activation='sigmoid')
])
E.summary()
opt = Adam(0.0001)

for e in range(10):
    fte = E.predict(xte)
    vis_feature(fte, yte, 'vis-epoch-{}.png'.format(e))
    for i,(batchx1,batchx2,batchy1) in enumerate(triplet):
        with tf.GradientTape() as tape:
            fx1 = E(batchx1)
            fx2 = E(batchx2)
            fy1 = E(batchy1)
            d1 = tf.reduce_mean(dist(fx1,fx2))
            d2 = tf.reduce_mean(dist(fx1,fy1))
            L = tf.maximum(d1 - d2 + 2., 0)
        
        grad = tape.gradient(L, E.variables)
        opt.apply_gradients(zip(grad, E.variables))

        if i%100==0:
            msg = 'epoch: {}-{}, d1: {}, d2: {}, L: {}'.format(e, i, d1.numpy(), d2.numpy(), L.numpy())
            print('\r'+msg, end='')
    print()

# 训练后可视化一次
fte = E.predict(xte)
vis_feature(fte, yte, 'vis-epoch-end.png')

