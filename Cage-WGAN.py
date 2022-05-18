#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K
from keras.utils import plot_model, np_utils
from keras import initializers
from PIL import Image  
from skimage import io

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[10]:


class WGAN():
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
 
        # 设置参数（文献中推荐的参数）
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)
 
        # 建立并编译critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
 
        # 建立生成器
        self.generator = self.build_generator()
 
        # 生成器以噪声为输入，生成图像
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
 
        # 对wgan只训练生成器
        self.critic.trainable = False
 
        # critic以生成图像作为输入并转化为正确性
        valid = self.critic(img)
 
        # 建立并编译wgan混合模型
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
 
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
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
 
    def build_generator(self):
 
        model = Sequential()
 
        model.add(Dense(1024 * 4 * 4, activation="relu", input_dim=self.z_dim))
        model.add(Reshape((4, 4, 1024)))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
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
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
 
        model.summary()
 
        img = Input(shape=self.img_shape)
        validity = model(img)
 
        return Model(img, validity)
 
    def train(self, iterations, batch_size=64, sample_interval=100):
 
        # 加载数据集
        input_dir = "data/cage/"
        X_train = self.prepare_data(input_dir)
 
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
 
        for iteration in range(iterations):
 
            for _ in range(self.n_critic):
 
                # ---------------------
                #  训练鉴别器
                # ---------------------
 
                # 抽取一批样本
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # 抽取随机变量作为生成器的输入
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
 
                # 生成一批新图像
                gen_imgs = self.generator.predict(noise)
 
                # 训练critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
 
                # 修建critic权重
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
 
            # ---------------------
            #  训练生成器
            # ---------------------
 
            g_loss = self.combined.train_on_batch(noise, valid)
 
            # 保存生成的图像样本
            if (iteration + 1) % sample_interval == 0:
                # 过程输出
                print ("%d [D loss: %f acc.: %.2f%%] [G loss: %f]" % (iteration + 1, 1 - d_loss[0], 100*d_loss[1], g_loss[0]))
                
                self.sample_images(iteration + 1)
                
            if (iteration + 1) % (sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((1 - d_loss[0], 1 - g_loss[0]))
                self.accuracies.append(100*d_loss[1])
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
        
        accuracies = np.array(self.accuracies)
        # 绘制鉴别器的准确度图像
        plt.figure(figsize=(10,5))
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
        gen_imgs = 0.5 * gen_imgs + 0.5
 
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        
        if not os.path.exists('cage_images-WGAN/'):
            os.makedirs('cage_images-WGAN/')
        fig.savefig("cage_images-WGAN/cage_%d.png" % iteration)
        plt.close()


# In[11]:


if __name__ == '__main__':
    wgan = WGAN()
    gen_imgs = wgan.train(iterations=30000, batch_size=64, sample_interval=100)
    


# # IS

# In[13]:


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




