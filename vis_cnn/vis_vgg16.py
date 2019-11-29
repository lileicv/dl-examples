''' 可视化 CNN 类激活 map
'''

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--impath', type=str, help='可视化哪张图片')
args = parser.parse_args()

impath = args.impath

# 加载原始图像
img = cv2.imread(impath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))

imtensor = np.expand_dims(img, 0)
imtensor = preprocess_input(imtensor)
imtensor = tf.convert_to_tensor(imtensor)

# 加载模型直接分类
model = VGG16()

# 类激活map可视化
heatmap_model = Model([model.inputs], [model.output, model.get_layer('block5_conv3').output])
with tf.GradientTape() as g:
    pred, lastconv = heatmap_model(imtensor)
    print('分类结果：', decode_predictions(pred.numpy()))
    idx = np.argmax(pred[0])
    grads = g.gradient(pred[:, idx], lastconv)
    pooled_grads = K.mean(grads, axis=(0,1,2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, lastconv), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap = heatmap / max_heat * 255
heatmap = heatmap.astype('uint8')
heatmap = cv2.resize(heatmap[0], (224, 224))

imname = impath.split('/')[-1].split('.')[0]
cv2.imwrite('./results/{}-heatmap.jpg'.format(imname), heatmap)
cv2.imwrite('./results/{}.jpg'.format(imname), img)



