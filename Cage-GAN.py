#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model, np_utils

from PIL import Image  
from skimage import io

import matplotlib.pyplot as plt

import sys

import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Suppress warnings because the warning Keras gives us about non-trainable parameters is by design:
# The Generator trainable parameters are intentionally held constant during Discriminator training and vice versa
import warnings; warnings.simplefilter('ignore')


# In[32]:


class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3

        #输入图像维度
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        #噪音向量，用于输入至生成器
        self.z_dim = 100
        
        #self.input_dir = "cage/"
        
        LEARNING_RATE = 0.0002
        TRAINING_RATIO = 1
        BETA_1 = 0.0
        BETA_2 = 0.9
        BN_MIMENTUM = 0.1
        BN_EPSILON  = 0.00002
        
        self.losses = []
        self.accuracies = []
        # 添加循环检查点
        self.iteration_checkpoints = []
        
        # 创建和编译鉴别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), metrics=['accuracy'])

        # 创建生成器
        self.generator = self.build_generator()
        
        # 生成器以噪声为输入，生成图像
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        # 训练生成器时保持鉴别器参数不变
        self.discriminator.trainable = False
        
        # 鉴别器以生成图像作为输入并转化为准确性
        validity = self.discriminator(img)

        # 创建和编译GAN
        self.gan = Model(z, validity)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2))
        
    # 读取数据的函数
    def prepare_data(self, input_dir):
        '''
        函数功能：通过输入图像的路径，读取训练数据 
        :return: 返回读取好的训练数据
        '''
        tf.compat.v1.disable_eager_execution()
 
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
 
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        data = tf.reshape(data, [-1, self.img_rows, self.img_cols, self.channels])
        train_data = data * 1.0 / 127.5 - 1.0                         #对data进行正则化
        X_train = sess.run(train_data)
        sess.close()
        return X_train
    
    def build_generator(self):
    
        model = Sequential()
    
        # 隐藏层-全连接层
        model.add(Dense(128, input_dim=self.z_dim))

        # Leaky ReLU激活函数
        model.add(LeakyReLU(alpha=0.01))

        # 输出层，使用tanh激活函数
        model.add(Dense(64*64*3, activation='tanh'))
    
        #将生成器输出数据重构为图像尺寸
        model.add(Reshape(self.img_shape))

        z = Input(shape=(self.z_dim,))
        img = model(z)

        return Model(z, img)
    
    def build_discriminator(self):
    
        model = Sequential()
    
        #将图像数据转化为一维向量
        model.add(Flatten(input_shape=self.img_shape))

        # 隐藏层-全连接层
        model.add(Dense(128))

        # Leaky ReLU激活函数
        model.add(LeakyReLU(alpha=0.01))
    
        # 输出层，使用sigmoid激活函数
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        prediction = model(img)

        return Model(img, prediction)
    
    def train(self, iterations, batch_size, sample_interval):
        
        input_dir = "data/cage/"
        
        X_train = self.prepare_data(input_dir)
        
        # 真实图像标签标记为1，虚假图像标签标记为0
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
        
        # -------------------------
        #  训练鉴别器
        # -------------------------

            # 随机获取一批真实图像
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 生成一批虚假图像
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # 训练鉴别器
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  训练生成器
        # ---------------------

            # 生成一批虚假图像
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # 训练生成器
            g_loss = self.gan.train_on_batch(z, real)
            
            if (iteration + 1) % sample_interval == 0:
            
                # 输出训练过程数据
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration + 1, d_loss[0], 100*d_loss[1], g_loss))

                # 输出生成图像样本 
                self.sample_images(iteration + 1)
                
            if (iteration + 1) % (sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((d_loss[0], g_loss))
                self.accuracies.append(100*d_loss[1])
                self.iteration_checkpoints.append(iteration + 1)
            
        #随机获取真实图像
        cnt = 0
        image_grid_rows, image_grid_columns = 4, 4
        idx_rand = np.random.randint(0, X_train.shape[0], image_grid_rows * image_grid_columns)
        imgs_rand = X_train[idx_rand]
    
        # 绘图设置
        fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                    figsize=(4,4), sharey=True, sharex=True)

        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                axs[i,j].imshow(imgs_rand[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
    
        plt.suptitle("Real Images")
        plt.show()
        
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
        
        return gen_imgs, X_train
        
    def sample_images(self, iteration):
        
        image_grid_rows=4
        image_grid_columns=4
        # 随机噪声样本
        z = np.random.normal(0, 1, 
              (image_grid_rows * image_grid_columns, self.z_dim))

        # 根据随机噪声生成图像 
        gen_imgs = self.generator.predict(z)

        # 缩放范围到[0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
    
        # 绘图设置
        fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                    figsize=(4,4), sharey=True, sharex=True)
    
        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
                
        # 保存图像
        if not os.path.exists('cage_images-GAN/'):
            os.makedirs('cage_images-GAN/')
        fig.savefig("cage_images-GAN/cage_%d.png" % iteration)
        plt.close()
        


# In[33]:


if __name__ == '__main__':
    gan = GAN()
    gen_imgs, X_train = gan.train(iterations=30000, batch_size=64, sample_interval=100)


# # IS

# In[34]:


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


# In[3]:


img_rows = 64
img_cols = 64
channels = 3
def prepare_data(input_dir):
    '''
    函数功能：通过输入图像的路径，读取训练数据 
    :return: 返回读取好的训练数据
    '''
    tf.compat.v1.disable_eager_execution()
 
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
 
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    data = tf.reshape(data, [-1, img_rows, img_cols, channels])
    train_data = data * 1.0 / 127.5 - 1.0                         #对data进行正则化
    X_train = sess.run(train_data)
    sess.close()
    return X_train

input_dir = "data/cage/"
X_train = prepare_data(input_dir)


# # FID

# In[4]:


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
images1 = X_train
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




