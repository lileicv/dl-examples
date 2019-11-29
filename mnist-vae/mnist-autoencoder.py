'''
mnist autoencoder
'''
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

# Build Model
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(10),
    Dense(7*7*32),
    Reshape((7,7,32)),
    UpSampling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    UpSampling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    Conv2D(1, 3, padding='same', activation='sigmoid')
])

x = Input(shape=(28,28,1))
y = model(x)
model = Model(x,y)

def loss_mse_wrapper(xin):
    def loss_mse(ytrue, ypred):
        return K.mean(K.square(xin-ypred))
    return loss_mse
model.compile(loss=loss_mse_wrapper(x), optimizer='adam')

# Load Dataset
(xtr,_),(xte,_) = mnist.load_data()
xtr = np.expand_dims(xtr/255.,3)
xte = np.expand_dims(xte/255.,3)
model.fit(xtr,xtr,epochs=10,batch_size=64,validation_data=(xte,xte), shuffle=True)

