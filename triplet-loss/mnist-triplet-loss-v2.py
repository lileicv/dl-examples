'''
tensorflow-addons 提供了 triplet loss， semi-hard 在线训练的方式
'''
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.datasets import mnist
from sklearn.manifold import TSNE
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

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
xtr = np.expand_dims(xtr, -1).astype('float32')
xte = np.expand_dims(xte, -1).astype('float32')

print(xtr.shape, ytr.shape)
print(xte.shape, yte.shape)
trset = tf.data.Dataset.from_tensor_slices((xtr,ytr)).batch(50)
teset = tf.data.Dataset.from_tensor_slices((xte,yte)).batch(50)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss())
#model.fit(xtr,ytr,epochs=10, batch_size=100)
model.fit(trset.repeat(), epochs=10, steps_per_epoch=xtr.shape[0]//50)

fte = model.predict(xte)
vis_feature(fte,yte,'./vis.png')
