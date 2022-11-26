# -*- coding: utf-8 -*-
"""
Created on Thu May 28 00:25:58 2020

@author: Vivan
"""

# importing necessary libraries and modules

from keras.layers import Dropout, Conv2D, Input, Dense, Flatten, LeakyReLU, Conv2DTranspose, BatchNormalization, Reshape
from keras.models import Model
import numpy as np
from keras.datasets import mnist
import keras
from keras.utils import plot_model
from matplotlib import pyplot as plt

# designing the generator model

genInput = Input((100,))

dense11 = Dense(7*7*256)(genInput)
bat11 = BatchNormalization()(dense11)
leak11 = LeakyReLU()(bat11)
drop11 = Dropout(0.3)(leak11)

resh = Reshape((7,7,256))(drop11)

dicon12 = Conv2DTranspose(128, 5, strides = 1, padding = 'same')(resh)
bat12 = BatchNormalization()(dicon12)
leak12 = LeakyReLU()(bat12)
drop12 = Dropout(0.3)(leak12)

dicon13 = Conv2DTranspose(64, 5, strides = 2, padding = 'same')(drop12)
bat13 = BatchNormalization()(dicon13)
leak13 = LeakyReLU()(bat13)
drop13 = Dropout(0.3)(leak13)

genOutput = Conv2DTranspose(1, 5, strides = 2, activation = 'tanh', padding = 'same')(drop13)

# making the generator model

gener = Model(genInput, genOutput)

# com[piling the generator model
 
gener.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy())

# designing the discriminator model

disInput = Input((28,28,1))

conv21 = Conv2D(64, 5, strides = 2, padding = 'same')(disInput)
leak21 = LeakyReLU()(conv21)
drop21 = Dropout(0.3)(leak21)

conv22 = Conv2D(128, 5, strides = 2, padding = 'same')(drop21)
leak22 = LeakyReLU()(conv22)
drop22 = Dropout(0.3)(leak22)

conv23 = Conv2D(256, 3, strides = 2, padding = 'same')(drop22)
leak23 = LeakyReLU()(conv23)
drop23 = Dropout(0.3)(leak23)

flat = Flatten()(drop23)
disOutput = Dense(1, activation = 'sigmoid')(flat)

# making the discriminator model
 
discr = Model(disInput, disOutput)

# compiling the discriminator model

discr.trainable = True
discr.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy())

# freezing the discriminator weights

discr.trainable = False

# making the GAN model

gan = Model(inputs = gener.input, outputs = discr(gener.output))

# compilimg the GAN model

gan.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy())

# plotting all 3 models

plot_model(gan, to_file = 'gan.pdf', show_shapes = True)
plot_model(gener, to_file = 'generator.pdf', show_shapes = True)
plot_model(discr, to_file = 'discriminator.pdf', show_shapes = True)

# loading the mnist dataset

(train_images, _ ) , (test_images, _ ) = mnist.load_data()

# reshaping the training images of the dataset

train_images = train_images.reshape((60000,28,28,1)) 

# normalizing the training dataset

train_images = (train_images - 127.5) / 127.5

# defining the noise generator function

def generate_noise():
    n = np.random.normal(0, 1, size = (256*100))
    n = n.reshape(256, 100)
    return n

# defining the random real sample generator function

def sample_generator(dataset, n_samples):
    number = np.random.randint(0, dataset.shape[0], n_samples)
    xt = dataset[number]
    return xt

# defining the discriminator training function

def train_the_discriminator():
    x1 = sample_generator(train_images, 256)
    y1 = np.ones(256) 
    y2 = np.zeros(256)
    
    noise = generate_noise()
    
    x2 = gener.predict(noise)

    discr.train_on_batch(x1, y1)
    discr.train_on_batch(x2, y2)

# defining the GAN training function
        
def train_the_gan():

    y1 = np.ones(256) 
    noise = generate_noise()
    l = gan.train_on_batch(noise, y1)
    return l

# defining the test noise generator function

def test_noise():
    n = np.random.normal(0, 1, size = (64*100))
    n = n.reshape((64,100)) 
    return n
 # defining the generator outlet function
 
def test_image():
    noise1 = test_noise()
    result = gener.predict(noise1)
    plt.figure(figsize=(8,8))

    for j in range(result.shape[0]):
        plt.subplot(8, 8, j+1)
        plt.imshow((result[j, :, :, 0] * 127.5) + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
    
    
for i in range(70001):
    if i % 1000 == 0:
        test_image()
    train_the_discriminator()
        
    loss = train_the_gan()
        
    print(i)
    print(loss)




