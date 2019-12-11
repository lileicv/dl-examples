'''
rejection sampling

有一点需要注意：原始数据集必须是 .repeat() 之后的
'''

import tensorflow as tf
import numpy as np


# tensorflow 类别不平衡之 rejection_sampling
x = [i for i in range(93)] + ['a', 'b', 'c', 'd', 'e', 'f', 'g']
y = [0 for i in range(93)] + [1 for i in range(7)]
x,y = np.array(x), np.array(y)
ds1 = tf.data.Dataset.from_tensor_slices((x,y)).repeat().batch(3)

resampler = tf.data.experimental.rejection_resample(lambda x,y:y, target_dist=[0.5, 0.5])
ds2 = ds1.unbatch().apply(resampler).batch(4)
ds2 = ds2.map(lambda y2, xy:xy)

for batchx, batchy in ds2.take(20):
    print(batchx.numpy(), batchy.numpy())
