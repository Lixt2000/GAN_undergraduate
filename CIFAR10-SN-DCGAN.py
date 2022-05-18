#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose, Activation, Reshape, LeakyReLU, Flatten, BatchNormalization
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.generic_utils import Progbar
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


# for resist GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


# In[3]:


from keras.datasets import cifar10

(X_train, y_train),(X_test,y_test) = cifar10.load_data()


# In[4]:


X = np.concatenate((X_test,X_train))
X.shape


# In[5]:


plt.imshow(X[9487])


# In[6]:


#Hyperperemeter
BATCHSIZE=64
LEARNING_RATE = 0.0002
TRAINING_RATIO = 1
BETA_1 = 0.0
BETA_2 = 0.9
EPOCHS = 500
BN_MIMENTUM = 0.1
BN_EPSILON  = 0.00002
SAVE_DIR = 'cifar10_images-SN-DCGAN/'

GENERATE_ROW_NUM = 8
GENERATE_BATCHSIZE = GENERATE_ROW_NUM*GENERATE_ROW_NUM


# In[7]:


def BuildGenerator(summary=True):
    model = Sequential()
    model.add(Dense(4*4*512, kernel_initializer='glorot_uniform' , input_dim=128))
    model.add(Reshape((4,4,512)))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MIMENTUM))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MIMENTUM))
    model.add(Conv2DTranspose(64,  kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MIMENTUM))
    model.add(Conv2DTranspose(3,   kernel_size=3, strides=1, padding='same', activation='tanh'))
    if summary:
        print("Generator")
        model.summary()
    return model


# In[8]:


def BuildDiscriminator(summary=True, spectral_normalization=True):
    if spectral_normalization:
        model = Sequential()
        model.add(ConvSN2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same', input_shape=(32,32,3) ))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(64, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(128, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(128, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(256, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(ConvSN2D(512, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Flatten())
        model.add(DenseSN(1,kernel_initializer='glorot_uniform'))
    else:
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same', input_shape=(32,32,3) ))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(64, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(128, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(128, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(256, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Conv2D(512, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
        model.add(LeakyReLU(0.1))
        model.add(Flatten())
        model.add(Dense(1,kernel_initializer='glorot_uniform'))
    if summary:
        print('Discriminator')
        print('Spectral Normalization: {}'.format(spectral_normalization))
        model.summary()
    return model


# In[9]:


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


# In[10]:


generator = BuildGenerator()
discriminator = BuildDiscriminator()


# In[11]:


Noise_input_for_training_generator = Input(shape=(128,))
Generated_image                    = generator(Noise_input_for_training_generator)
Discriminator_output               = discriminator(Generated_image)
model_for_training_generator       = Model(Noise_input_for_training_generator, Discriminator_output)
print("model_for_training_generator")
model_for_training_generator.summary()


# In[12]:


discriminator.trainable = False
model_for_training_generator.summary()


# In[13]:



model_for_training_generator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=wasserstein_loss)


# In[14]:


Real_image                             = Input(shape=(32,32,3))
Noise_input_for_training_discriminator = Input(shape=(128,))
Fake_image                             = generator(Noise_input_for_training_discriminator)
Discriminator_output_for_real          = discriminator(Real_image)
Discriminator_output_for_fake          = discriminator(Fake_image)

model_for_training_discriminator       = Model([Real_image,
                                                Noise_input_for_training_discriminator],
                                               [Discriminator_output_for_real,
                                                Discriminator_output_for_fake])
print("model_for_training_discriminator")
generator.trainable = False
discriminator.trainable = True
model_for_training_discriminator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=[wasserstein_loss, wasserstein_loss])
model_for_training_discriminator.summary()


# In[15]:


real_y = np.ones((BATCHSIZE, 1), dtype=np.float32)
fake_y = -real_y


# In[16]:


X = X/255*2-1


# In[17]:


plt.imshow((X[8787]+1)/2)


# In[18]:


test_noise = np.random.randn(GENERATE_BATCHSIZE, 128)
W_loss = []
discriminator_loss = []
generator_loss = []
for epoch in range(EPOCHS):
    np.random.shuffle(X)
    
    print("epoch {} of {}".format(epoch+1, EPOCHS))
    num_batches = int(X.shape[0] // BATCHSIZE)
    
    print("number of batches: {}".format(int(X.shape[0] // (BATCHSIZE))))
    
    progress_bar = Progbar(target=int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO)))
    minibatches_size = BATCHSIZE * TRAINING_RATIO
    
    start_time = time()
    for index in range(int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO))):
        progress_bar.update(index)
        discriminator_minibatches = X[index * minibatches_size:(index + 1) * minibatches_size]
        
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCHSIZE : (j + 1) * BATCHSIZE]
            noise = np.random.randn(BATCHSIZE, 128).astype(np.float32)
            discriminator.trainable = True
            generator.trainable = False
            discriminator_loss.append(model_for_training_discriminator.train_on_batch([image_batch, noise],
                                                                                      [real_y, fake_y]))
        discriminator.trainable = False
        generator.trainable = True
        generator_loss.append(model_for_training_generator.train_on_batch(np.random.randn(BATCHSIZE, 128), real_y))
    
    print('\nepoch time: {}'.format(time()-start_time))
    
    W_real = model_for_training_generator.evaluate(test_noise, real_y)
    print(W_real)
    W_fake = model_for_training_generator.evaluate(test_noise, fake_y)
    print(W_fake)
    W_l = W_real+W_fake
    print('wasserstein_loss: {}'.format(W_l))
    W_loss.append(W_l)
    #Generate image
    generated_image = generator.predict(test_noise)
    generated_image = (generated_image+1)/2
    for i in range(GENERATE_ROW_NUM):
        new = generated_image[i*GENERATE_ROW_NUM:i*GENERATE_ROW_NUM+GENERATE_ROW_NUM].reshape(32*GENERATE_ROW_NUM,32,3)
        if i!=0:
            old = np.concatenate((old,new),axis=1)
        else:
            old = new
    print('plot generated_image')
    # 缩放范围到[0, 1]
    old = 0.5 * old + 0.5
    if not os.path.exists('img/generated_img_CIFAR10_SN-DCGAN/'):
        os.makedirs('img/generated_img_CIFAR10_SN-DCGAN/')
    plt.imsave('{}/SN_epoch_{}.png'.format(SAVE_DIR, epoch+1), old)


# In[19]:


plt.plot(W_loss)


# In[20]:


plt.imshow(old)


# In[21]:


#Generate image
noise = np.random.randn(5000, 128).astype(np.float32)
gen_imgs = generator.predict(noise)
gen_imgs = (gen_imgs+1)/2


# # IS

# In[22]:


from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std
 
# load generated images
images = gen_imgs

# shuffle images
shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)


# # FID

# In[ ]:


import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# load generated and real images
images2 = gen_imgs
(images1, _), (_, _) = cifar10.load_data()
shuffle(images1)
images1 = images1[:5000]
print('Loaded', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)


# In[ ]:




