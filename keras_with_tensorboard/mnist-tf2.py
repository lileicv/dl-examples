'''
Eager 模式训练 Mnist
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd

# Global Config
epochs = 1
batchsize = 128

# Load Dataset
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = np.expand_dims(xtr.astype(np.float32),3)/255.
xte = np.expand_dims(xte.astype(np.float32),3)/255.
ytr = ytr.astype(np.int32)
yte = yte.astype(np.int32)
trset = tf.data.Dataset.from_tensor_slices((xtr, ytr)).batch(batchsize)
teset = tf.data.Dataset.from_tensor_slices((xte, yte)).batch(batchsize)

# Build Model
model = Sequential([
    L.Conv2D(32, 3, 1, 'same', activation='relu', input_shape=[28, 28, 1]),
    L.MaxPooling2D(),
    L.Conv2D(32, 3, 1, 'same', activation='relu'),
    L.MaxPooling2D(),
    L.Flatten(),
    L.Dense(128, activation='relu'),
    L.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
tb_callback = TensorBoard('./logs', write_graph=False)
#model.fit(trset, epochs=10)
hist = model.fit(trset, epochs=epochs, validation_data=teset, callbacks=[tb_callback])
#model.fit(xtr,ytr, epochs=epochs, batch_size=128, validation_data=[xte,yte], callbacks=[tb_callback])

fit_hist = pd.DataFrame(hist.history)
fit_hist.to_csv('a.csv', index=False)

