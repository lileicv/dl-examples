import tensorflow as tf
from tensorflow import keras

def res_block(x, filters, kernelsize, stride, downsample):
    ''' Resnet18 有4个block，每个block有2个卷积层
    '''
    h = L.



# for 18 or 34 layers
class Basic_Block(keras.Model):

    def __init__(self, filters, downsample=False, stride=1, block_name=''):
        self.expasion = 1
        super(Basic_Block, self).__init__()

        self.downsample = downsample

        self.conv2a = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          strides=stride,
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          )
        self.bn2a = keras.layers.BatchNormalization(axis=-1)

        self.conv2b = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal'
                                          )
        self.bn2b = keras.layers.BatchNormalization(axis=-1)

        self.relu = keras.layers.ReLU()

        if self.downsample:
            self.conv_shortcut = keras.layers.Conv2D(filters=filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     )
            self.bn_shortcut = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        print(inputs.shape)
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = self.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs
        
        print(x.shape)
        print(shortcut.shape)
        x = keras.layers.add([x, shortcut])
        x = self.relu(x)

        return x


class ResNet(keras.Model):
    def __init__(self, block, layers, num_classes=1000, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = keras.layers.Conv2D(filters=64,
                                         kernel_size=7,
                                         strides=2,
                                         kernel_initializer='glorot_uniform',
                                         name='conv1')
        self.bn_conv1 = keras.layers.BatchNormalization(axis=3, name='bn_conv1')
        self.max_pool = keras.layers.MaxPooling2D((3, 3),
                                                  strides=2,
                                                  padding='same')
        self.avgpool = keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.fc = keras.layers.Dense(num_classes, activation='softmax', name='result')

        # layer2
        self.res2 = self.mid_layer(block, 64, layers[0], stride=1, layer_number=2)

        # layer3
        self.res3 = self.mid_layer(block, 128, layers[1], stride=2, layer_number=3)

        # layer4
        self.res4 = self.mid_layer(block, 256, layers[2], stride=2, layer_number=4)

        # layer5
        self.res5 = self.mid_layer(block, 512, layers[3], stride=2, layer_number=5)

    def mid_layer(self, block, filter, block_layers, stride=1, layer_number=1):
        layer = keras.Sequential()
        if stride != 1 or filter * 4 != 64:
            layer.add(block(filters=filter,
                            downsample=True, stride=stride,
                            block_name='{}a'.format(layer_number)))

        for i in range(1, block_layers):
            p = chr(i + ord('a'))
            layer.add(block(filters=filter,
                            block_name='{}'.format(layer_number) + p))

        return layer

    def call(self, inputs, **kwargs):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        # layer2
        x = self.res2(x)
        # layer3
        x = self.res3(x)
        # layer4
        x = self.res4(x)
        # layer5
        x = self.res5(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def resnet18():
    return ResNet(Basic_Block, [2, 2, 2, 2], num_classes=1000)


if __name__ == '__main__':
    model = resnet18()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    model.save('a.h5')
