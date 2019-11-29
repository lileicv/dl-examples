'''
使用 vgg16 分类 cifar100
感谢钟雷师弟，帮忙调通代码
学习率设置大了，所以不收敛
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.datasets import cifar100, mnist
from tensorflow.keras.optimizers import Adam

class Imap:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def map_tr(self, path, label):
        ''' 这里输入的 path 有两种情况，要加以判断
            - 字符串，表示图片在磁盘上的地址，比如 miniImageNet 数据集，数据以图片的文件的形式存储在磁盘上
            - tensor，表示图片上的像素矩阵，比如 cifar100，直接把图片全部加载进来了
        '''
        if path.dtype == 'uint8':
            img = path
        else:
            raw = tf.io.read_file(path)
            img = tf.io.decode_jpeg(raw)
        img = tf.image.resize(img, [self.input_shape[0]*5//4, self.input_shape[1]*5//4])
        img = tf.image.random_crop(img, self.input_shape)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=63)
        #img = tf.image.random_contrast(img, lower=0.2, upper=1.8) # 加上这个增强，训练就不收敛了
        img = tf.cast(img, 'float32')/255.
        return img, label

    def map_te(self, path, label):
        if path.dtype == 'uint8':
            img = path
        else:
            raw = tf.io.read_file(path)
            img = tf.io.decode_jpeg(raw)
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = tf.cast(img, 'float32')/255.
        return img, label

def new_model(modelname, input_shape, nclass):
    ''' 使用 keras 自带的 vgg16 不好训练
    '''
    # build model
    if modelname == 'vgg16':
        basemodel = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    elif modelname == 'mobilenet_v2':
        basemodel = mobilenet_v2.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    model = Sequential([
        basemodel,
        L.Flatten(),
        L.Dense(1024, activation='relu', name='feat'),
        L.Dropout(0), # 这里不能dropout太多，否则模型不太容易训练
        L.Dense(nclass, activation='softmax')
    ])
    return model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load Model
model = new_model('vgg16', [224,224,3], 100)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# Load Dataset
(xtr, ytr), (xte, yte) = cifar100.load_data()
perm = np.random.permutation(len(xtr))
xtr = xtr[perm[:10000]]
ytr = ytr[perm[:10000]]
imap = Imap([224, 224, 3])

trset = tf.data.Dataset.from_tensor_slices((xtr,ytr)).map(imap.map_tr).batch(32).shuffle(10)
teset = tf.data.Dataset.from_tensor_slices((xte,yte)).map(imap.map_te).batch(32)
#model.fit(xtr,ytr, epochs=5, batch_size=64, validation_data=[xte,yte])
model.fit(trset, epochs=3, validation_data=teset)


# model.save('./model.h5')

# model = load_model('./model.h5')
# model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
# model.evaluate(xte, yte)
# feat_model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
# feat_model.summary()
# ftr = feat_model.predict(xtr)
# print(ftr.shape)

