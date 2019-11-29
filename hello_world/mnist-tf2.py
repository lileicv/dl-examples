'''
Eager 模式训练 Mnist
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

#tf.enable_eager_execution()

# Global Config
epochs = 5
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
optimizer = tf.keras.optimizers.Adam(0.0001)

# Traing the model
for e in range(epochs):
    print('Epoch {}'.format(e))
    for i, (batchx, batchy) in enumerate(trset):
        with tf.GradientTape() as tape:
            logits = model(batchx)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batchy)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        if i % 100 == 0:
            print('\r {}, loss {}'.format(i, loss), end=' ')
    acces = []
    for i, (batchx, batchy) in enumerate(teset):
        logits = model(batchx)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, -1, output_type=tf.int32), batchy), tf.float32))
        acces.append(acc)
    print('accuracy', np.mean(acces))
    tf.train.Checkpoint(optimizer=optimizer, model=model).save('./models/model')        

tf.train.Checkpoint(optimizer=optimizer, model=model).restore(tf.train.latest_checkpoint('./models/model'))

