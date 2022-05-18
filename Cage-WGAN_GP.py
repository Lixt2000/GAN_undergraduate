#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

get_ipython().run_line_magic('matplotlib', 'inline')

from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
import keras.backend as K
from keras.utils import plot_model, np_utils
from keras import initializers
from PIL import Image  
from skimage import io

import matplotlib.pyplot as plt

import sys

import numpy as np

import tensorflow as tf 

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


class RandomWeightedAverage(_Merge):
    """给真实样本和生成样本提供一个随机权重平均"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64,64,64,3))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# In[3]:


class WGANGP():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        
        self.losses = []
        self.accuracies = []
        # 添加循环检查点
        self.iteration_checkpoints = []

        # 参照文献中的参数与优化器设置
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # 建立生成器和评论器
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # 训练评论器时生成器不变
        self.generator.trainable = False

        # 输入真实图像
        real_img = Input(shape=self.img_shape)

        # 输入噪声
        z_disc = Input(shape=(self.z_dim,))
        # 基于噪声生成虚假图像样本
        fake_img = self.generator(z_disc)

        # 评论器对真实样本和虚假样本定义准确性
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # 建立真实样本和虚假样本的权重平均
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # 定义权重样本的准确性
        validity_interpolated = self.critic(interpolated_img)

        # 使用Python partial提供损失函数
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # 训练生成器时评论器不变
        self.critic.trainable = False
        self.generator.trainable = True

        # 向生成器中输入噪声样本
        z_gen = Input(shape=(self.z_dim,))
        # 基于样本生成样本
        img = self.generator(z_gen)
        # 判断样本准确性
        valid = self.critic(img)
        # 定义生成器
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    # 读取数据的函数
    def prepare_data(self, input_dir):
        '''
        函数功能：通过输入图像的路径，读取训练数据 
        :return: 返回读取好的训练数据
        '''
        
        # 遍历图像路径，并获取图像数量
        images = os.listdir(input_dir)    
        image_len = len(images)
 
        # 设置一个空data，用于存放数据
        data = np.empty((image_len, self.img_rows, self.img_cols, self.channels), dtype="float32")
 
        # 逐个图像读取
        for i in range(image_len):
            #如果导入的是skimage.io，则读取影像应该写为img = io.imread(input_dir + images[i])
            img = Image.open(input_dir + images[i])    #打开图像
            img = img.resize((self.img_rows, self.img_cols))             #将256*256变成64*64
            arr = np.asarray(img, dtype="float32")                    #将格式改为np.array
            data[i, :, :, :] = arr                                    #将其放入data中
 
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        data = tf.reshape(data, [-1, self.img_rows, self.img_cols, self.channels])
        train_data = data * 1.0 / 127.5 - 1.0                         #对data进行正则化
        X_train = sess.run(train_data)
        sess.close()
        return X_train
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        基于预测和加权后的真实/虚假样本，计算梯度惩罚
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # 开方计算欧式距离模长 ...
        gradients_sqr = K.square(gradients)
        #   ... 各行相加 ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... 开方
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # 对每一个单一样本计算 lambda * (1 - ||grad||)^2
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # 返回均值作为所有批量样本的损失
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
 
        model.add(Dense(1024 * 4 * 4, activation="relu", input_dim=self.z_dim))
        model.add(Reshape((4, 4, 1024)))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.z_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iterations, batch_size, sample_interval=100):
        
        # 加载数据集
        input_dir = "data/cage/"
        X_train = self.prepare_data(input_dir)
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for iteration in range(iterations):

            for _ in range(self.n_critic):

                # ---------------------
                #  训练评论器
                # ---------------------

                # 抽取随机真实样本
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # 抽取噪声，输入生成器
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
                # 训练评论器
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  训练生成器
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # 保存生成的图像样本
            if (iteration + 1) % sample_interval == 0:
                # 过程输出
                print ("%d [D loss: %f] [G loss: %f]" % (iteration + 1, 1 - d_loss[0], g_loss))
                
                self.sample_images(iteration + 1)
                
            if (iteration + 1) % (sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((1 - d_loss[0], g_loss))
                #self.accuracies.append(100*d_loss[1])
                self.iteration_checkpoints.append(iteration + 1)
                
        losses = np.array(self.losses)
        # 绘制鉴别器和生成器的损失图像
        plt.figure(figsize=(10,5))
        plt.plot(self.iteration_checkpoints, losses.T[0], label="Discriminator Loss")
        plt.plot(self.iteration_checkpoints, losses.T[1], label="Generator Loss")
        plt.xticks(self.iteration_checkpoints, rotation=90)
        plt.title("Training Losses")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        
        #保存训练后的生成图像
        z = np.random.normal(0, 1, (5000, self.z_dim))
        gen_imgs = self.generator.predict(z)
        
        return gen_imgs

    def sample_images(self, iteration):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 重构范围到[0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(4,4), sharey=True, sharex=True)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        
        if not os.path.exists('cage_images-WGAN-GP/'):
            os.makedirs('cage_images-WGAN-GP/')
        fig.savefig("cage_images-WGAN-GP/cage_%d.png" % iteration)
        plt.close()


# In[4]:


if __name__ == '__main__':
    wgan = WGANGP()
    gen_imgs = wgan.train(iterations=5000, batch_size=64, sample_interval=100)


# # IS

# In[ ]:


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


img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
z_dim = 100
def prepare_data(input_dir):
        '''
        函数功能：通过输入图像的路径，读取训练数据 
        :return: 返回读取好的训练数据
        '''
 
        # 遍历图像路径，并获取图像数量
        images = os.listdir(input_dir)    
        image_len = len(images)
 
        # 设置一个空data，用于存放数据
        data = np.empty((image_len, img_rows, img_cols, channels), dtype="float32")
 
        # 逐个图像读取
        for i in range(image_len):
            #如果导入的是skimage.io，则读取影像应该写为img = io.imread(input_dir + images[i])
            img = Image.open(input_dir + images[i])    #打开图像
            img = img.resize((img_rows, img_cols))             #将256*256变成64*64
            arr = np.asarray(img, dtype="float32")                    #将格式改为np.array
            data[i, :, :, :] = arr                                    #将其放入data中
 
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        data = tf.reshape(data, [-1, img_rows, img_cols, channels])
        train_data = data * 1.0 / 127.5 - 1.0                         #对data进行正则化
        X_train = sess.run(train_data)
        sess.close()
        return X_train
input_dir = "data/cage/"
X_train = prepare_data(input_dir)


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
images2 = gen_imgs[:318]
images1 = X_train
shuffle(images1)
images1 = images1[:318]
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





# In[5]:


from __future__ import print_function, division

get_ipython().run_line_magic('matplotlib', 'inline')

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

import tensorflow as tf

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[6]:


def load_mnist():
    
    path = r'data/mnist.npz' #放置mnist.py的目录。注意斜杠
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


# In[ ]:


class RandomWeightedAverage(_Merge):
    """给真实样本和生成样本提供一个随机权重平均"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64,28,28,1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# In[ ]:


class WGANGPm():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        
        self.losses = []
        self.accuracies = []
        # 添加循环检查点
        self.iteration_checkpoints = []

        # 参照文献中的参数与优化器设置
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # 建立生成器和评论器
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # 训练评论器时生成器不变
        self.generator.trainable = False

        # 输入真实图像
        real_img = Input(shape=self.img_shape)

        # 输入噪声
        z_disc = Input(shape=(self.z_dim,))
        # 基于噪声生成虚假图像样本
        fake_img = self.generator(z_disc)

        # 评论器对真实样本和虚假样本定义准确性
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # 建立真实样本和虚假样本的权重平均
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # 定义权重样本的准确性
        validity_interpolated = self.critic(interpolated_img)

        # 使用Python partial提供损失函数
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # 训练生成器时评论器不变
        self.critic.trainable = False
        self.generator.trainable = True

        # 向生成器中输入噪声样本
        z_gen = Input(shape=(self.z_dim,))
        # 基于样本生成样本
        img = self.generator(z_gen)
        # 判断样本准确性
        valid = self.critic(img)
        # 定义生成器
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        基于预测和加权后的真实/虚假样本，计算梯度惩罚
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # 开方计算欧式距离模长 ...
        gradients_sqr = K.square(gradients)
        #   ... 各行相加 ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... 开方
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # 对每一个单一样本计算 lambda * (1 - ||grad||)^2
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # 返回均值作为所有批量样本的损失
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.z_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.z_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

         model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iterations, batch_size, sample_interval=100):

        # 加载数据
        (X_train, y_train), (x_test, y_test) = load_mnist()

        # 范围为(-1, 1)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for iteration in range(iterations):

            for _ in range(self.n_critic):

                # ---------------------
                #  训练评论器
                # ---------------------

                # 抽取随机真实样本
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # 抽取噪声，输入生成器
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
                # 训练评论器
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  训练生成器
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # 保存生成的图像样本
            if (iteration + 1) % sample_interval == 0:
                # 过程输出
                print ("%d [D loss: %f acc.: %.2f%%] [G loss: %f]" % (iteration + 1, 1 - d_loss[0], 100*d_loss[1], g_loss[0]))
                
                self.sample_images(iteration + 1)
                
            if (iteration + 1) % (sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((1 - d_loss[0], g_loss))
                self.accuracies.append(100*d_loss[1])
                self.iteration_checkpoints.append(iteration + 1)
                
        losses = np.array(self.losses)
        # 绘制鉴别器和生成器的损失图像
        plt.figure(figsize=(10,5))
        #plt.plot(losses.T[0], label="Discriminator Loss")
        #plt.plot(losses.T[1], label="Generator Loss")
        plt.plot(self.iteration_checkpoints, losses.T[0], label="Discriminator Loss")
        plt.plot(self.iteration_checkpoints, losses.T[1], label="Generator Loss")
        plt.xticks(self.iteration_checkpoints, rotation=90)
        plt.title("Training Losses")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        
        accuracies = np.array(self.accuracies)
        # 绘制鉴别器的准确度图像
        plt.figure(figsize=(10,5))
        #plt.plot(accuracies, label="Discriminator Accuracy")
        plt.plot(self.iteration_checkpoints, accuracies, label="Discriminator Accuracy")
        plt.xticks(self.iteration_checkpoints, rotation=90)
        plt.yticks(range(0, 100, 5))
        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        
        #保存训练后的生成图像
        z = np.random.normal(0, 1, (5000, self.z_dim))
        gen_imgs = self.generator.predict(z)
        
        return gen_imgs

    def sample_images(self, iteration):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 重构范围到[0,1]
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c, figsize=(4,4), sharey=True, sharex=True)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        if not os.path.exists('mnist_images/'):
            os.makedirs('mnist_images/')
        fig.savefig("mnist_images/mnist_%d.png" % iteration)
        plt.close()


# In[ ]:


if __name__ == '__main__':
    wgan = WGANGPm()
    gen_imgs = wgan.train(iterations=5000, batch_size=64, sample_interval=100)


# In[ ]:


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


# In[ ]:





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
(images1, _), (_, _) = load_mnist()
images1 = np.expand_dims(images1, axis=3)
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




