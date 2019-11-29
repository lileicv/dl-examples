'''
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar100

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

def res_block(x, filters, kernelsize, stride, downsample, name=''):
    ''' Resnet18有4个block，每个block有两个卷积层
    '''
    h = L.Conv2D(filters, kernelsize, stride, 'same', kernel_initializer='he_normal', name=name+'conv1')(x)
    h = L.BatchNormalization(axis=-1, name=name+'bn1')(h)
    h = L.ReLU(name=name+'relu1')(h)
    h = L.Conv2D(filters, kernelsize, 1, 'same', kernel_initializer='he_normal', name=name+'conv2')(h)
    h = L.BatchNormalization(axis=-1, name=name+'bn2')(h)
    h = L.ReLU(name=name+'relu2')(h)
    if downsample:
        shortcut = L.Conv2D(filters, 1, stride, 'same', kernel_initializer='he_normal', name=name+'sideconv')(x)
        shortcut = L.BatchNormalization(axis=-1, name=name+'side-relu')(shortcut)
    else:
        shortcut = x
    h = L.add([h, shortcut], name=name+'add')
    h = L.ReLU(name=name+'relu3')(h)
    return h

def resnet18(input_shape=[224,224,3], nbclass=100):
    x = L.Input(shape=input_shape)
    h = L.Conv2D(64, 7, 2, 'same', kernel_initializer='glorot_uniform', name='conv1')(x) # ==> 112
    h = L.BatchNormalization(axis=-1, name='conv1_bn')(h)
    h = L.ReLU()(h)
    h = L.MaxPooling2D((3,3), strides=2, padding='same')(h) # ==> 56
    
    h = res_block(h,  64, 3, 1, True, name='block1a')
    h = res_block(h,  64, 3, 1, False, name='block1b') # ==> 56
    h = res_block(h, 128, 3, 2, True, name='block2a')  # ==> 28
    h = res_block(h, 128, 3, 1, False, name='block2b') # ==> 28
    h = res_block(h, 256, 3, 2, True, name='block3a')
    h = res_block(h, 256, 3, 1, False, name='block3b')
    h = res_block(h, 512, 3, 2, True, name='block4a')
    h = res_block(h, 512, 3, 1, False, name='block4b')

    h = L.GlobalAveragePooling2D()(h)
    h = L.Dense(nbclass, activation='softmax')(h)
    model = Model(inputs=[x], outputs=[h])
    return model

if __name__=='__main__':
    model = resnet18([32,32,3], 100)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.save('model.h5')

    # Load Dataset
    (xtr, ytr), (xte, yte) = cifar100.load_data()
    perm = np.random.permutation(len(xtr))
    xtr = xtr[perm[:10000]]
    ytr = ytr[perm[:10000]]
    imap = Imap([32, 32, 3])

    trset = tf.data.Dataset.from_tensor_slices((xtr,ytr)).map(imap.map_tr).batch(32).shuffle(10)
    teset = tf.data.Dataset.from_tensor_slices((xte,yte)).map(imap.map_te).batch(32)
    model.fit(trset, epochs=3, validation_data=teset)

    #model.fit(xtr,ytr, epochs=5, batch_size=64, validation_data=[xte,yte])

