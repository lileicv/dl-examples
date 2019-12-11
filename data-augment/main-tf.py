'''
'''

import cv2
import tensorflow as tf
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

# 随机调整颜色
impath = '/home/lilei/data/cat.jpg'
img = cv2.imread(impath)
for i in range(10):
    im2 = tf.image.random_hue(img, max_delta=0.5)
    im2 = tf.image.random_saturation(im2, lower=0.2, upper=1.8)
    cv2.imwrite('./img-{}.jpg'.format(i), im2.numpy())




