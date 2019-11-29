'''
mnist autoencoder
'''
import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2D, UpSampling2D, Reshape, Flatten, Dense, Lambda
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy

def sampling(args):
    mean, log_var = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mean + K.exp(0.5 * log_var) * epsilon

def vae():
    xinput = Input(shape=(28,28,1))
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(xinput) # ==> [14,14,32]
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)      # ==> [7, 7, 64]
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    mean = Dense(2)(x)
    logvar = Dense(2)(x)
    z = Lambda(sampling, output_shape=(2,), name='sampling')([mean, logvar])
    #encoder = Model(xinput, [mean, logvar, z])

    #zinput = Input(shape=(2,))
    x = Dense(7*7*64)(z)
    x = Reshape([7,7,64])(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)         # ==> [7, 7, 64]
    x = UpSampling2D()(x)                                           # ==> [14,14,64]
    x = Conv2D(32, 3, padding='same', activation='relu')(x)         # ==> [14,14,32]
    x = UpSampling2D()(x)                                           # ==> [28,28,32]
    xoutput = Conv2D(1, 3, padding='same', activation='relu')(x)        # ==> [28,28,1]
    #decoder = Model(zinput, out)
    
    #xoutput = decoder(z)
    vae = Model(xinput, xoutput)

    rec_loss = binary_crossentropy(xinput, xoutput)
    rec_loss *= 28*28
    rec_loss = K.mean(rec_loss, axis=-1)
    rec_loss = K.mean(rec_loss, axis=-1)
    kl_loss = -0.5*K.sum(1+logvar-K.square(mean)-K.exp(logvar), axis=-1)

    vae_loss = K.mean(rec_loss+kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', metrics=['mse'])
    #vae.load_weights('a.h5') 
    vae.metrics_tensors.append(K.mean(K.abs(xinput-xoutput)))
    vae.metrics_names.append('mse')
    #vae.load_weights('a.h5')
    return vae

# Build Model
model = vae()
model.summary()

# Load Dataset
(xtr,_),(xte,_) = mnist.load_data()
xtr = np.expand_dims(xtr/255.,3)
xte = np.expand_dims(xte/255.,3)
model.fit(xtr, epochs=3,batch_size=64,shuffle=True)
model.save('a.h5')
