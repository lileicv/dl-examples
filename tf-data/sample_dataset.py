'''
sample dataset
有一点需要注意：原始数据集必须是 .repeat() 之后的
'''

import numpy as np
import tensorflow as tf

x = [1 for i in range(97)] + [2 for i in range(3)]

ds = tf.data.Dataset.from_tensor_slices((x,x,x))

ds1 = ds.filter(lambda x,y,z:x==1).repeat()
ds2 = ds.filter(lambda x,y,z:x==2).repeat()

ds3 = tf.data.experimental.sample_from_datasets([ds1, ds2], [0.5, 0.5]).batch(10)

for x, y, z in ds3.take(20):
    print(x.numpy())
