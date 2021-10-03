import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import os
from tensorflow.keras.layers import * 
from tensorflow.keras import * 
import tensorflow_addons as tfa

def downsample(filters, apply_batchnorm = True):

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2D(filters, 
        kernel_size = 4, 
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = not apply_batchnorm))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def upsample(filters, apply_dropout = False):

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2DTranspose(filters, 
        kernel_size = 4, 
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result

def ResidualBlock():

    inputs = tf.keras.layers.Input(shape=[None,None,256])

    result = Sequential()
    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2D(256, 
        kernel_size = 3, 
        strides = 1,
        padding = "same",
        kernel_initializer = initializer))

    result.add(BatchNormalization())

    result.add(ReLU())

    result.add(Conv2D(256, 
        kernel_size = 3, 
        strides = 1,
        padding = "same",
        kernel_initializer = initializer))

    result.add(BatchNormalization())

    x = inputs

    x = x + result(x)

    x = ReLU()(x)

    return Model(inputs = inputs, outputs = x)


def downsampleHD(filters, kernel_size, stride):

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2D(filters, 
        kernel_size = kernel_size, 
        strides = stride, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = False))

    result.add(tfa.layers.InstanceNormalization())

    result.add(ReLU())

    return result

def upsampleHD():

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2DTranspose(32, 
        kernel_size = 3, 
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = False))

    result.add(tfa.layers.InstanceNormalization())

    result.add(ReLU())

    return result

def GeneratorHD():

    inputs = tf.keras.layers.Input(shape=[None,None,3])

    initializer = tf.random_normal_initializer(0,0.02)

    x = inputs

    x = downsample(64, apply_batchnorm = False)(x) 
    x = downsample(128)(x)
    x = downsample(256)(x)

    for i in range(3):
        x = ResidualBlock()(x)

    x = upsample(256)(x)
    x = upsample(128)(x)
    x = upsample(64)(x)

    last = Conv2DTranspose(filters = 3, 
        kernel_size = 4,
        strides = 1, 
        padding = "same", 
        kernel_initializer = initializer,
        activation = "tanh")

    x = last(x)
    
    return Model(inputs = inputs, outputs = x)




def Generator():
    inputs = tf.keras.layers.Input(shape=[None,None,3])

    downstack = [
        downsample(64, apply_batchnorm = False),
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512)
    ]

    upstack = [
        upsample(512, apply_dropout = True),
        upsample(512, apply_dropout = True),
        upsample(512, apply_dropout = True),
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64)
    ]

    initializer = tf.random_normal_initializer(0,0.02)

    last = Conv2DTranspose(filters = 3, 
        kernel_size = 4,
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        activation = "tanh")

    x = inputs
    s = []

    for down in downstack:
        x = down(x)
        s.append(x)

    s = reversed(s[:-1])

    for up, sk in zip(upstack,s):
        x = up(x)
        x = Concatenate()([x, sk])

    x = last(x)

    return Model(inputs = inputs, outputs = x)

def Discriminator():
    ini = Input(shape=[None,None,3], name = "input_img")
    gen = Input(shape=[None,None,3], name = "gener_img")

    con = concatenate([ini,gen])

    initializer = tf.random_normal_initializer(0,0.02)

    down1 = downsample(64, apply_batchnorm = False)(con)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)
    down4 = downsample(512)(down3)
    
    last = Conv2D(filters = 1, 
        kernel_size = 4,
        strides = 1, 
        padding = "same", 
        kernel_initializer = initializer)(down4)

    return tf.keras.Model(inputs = [ini, gen], outputs = last)


