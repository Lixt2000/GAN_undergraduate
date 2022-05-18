#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from model_sphere_gan import Generator, Discriminator

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# In[2]:


class SphereGAN():
    def get_reference_point(self, coord=None):
        if coord == None:
            ref_p_np = np.zeros( (1,self.feature_dim+1 ) ).astype(np.float32)
            ref_p_np[0,self.feature_dim] = 1.0
            return tf.constant(ref_p_np)
        else:
            return coord
    def _dist_sphere(self, a,b,r):
        return tf.acos( tf.matmul(a , tf.transpose(b))) ** r
    def dist_weight_mode(self, r):
        if self.weight_mode == 'normalization':
            decayed_dist = ( (1/self.decay_ratio)*np.pi)**r
        elif self.weight_mode == 'half':
            decayed_dist = (np.pi)**r
        else:
            decayed_dist = 1
        return decayed_dist
    
    def eval_moments(self, y_true, y_pred):
        ref_p = self.get_reference_point()
        d = 0.0
        for r in range(1, self.moments + 1):
            d = d + self._dist_sphere(y_pred, ref_p, r) / self.dist_weight_mode(r)
        return K.mean(y_true * d)
    
    def __init__(self):
        self.img_shape = (32, 32, 3)
        self.batch_size = 64
        self.z_dim = 128
        self.feature_dim = 1024
        self.nb_learning = 1
        self.moments = 1 # [3] is suggested but [1] is enough.
        self.epochs = int(5E+5)
        self.sample_interval = 100
        self.weight_mode = None
        self.loss_mode = None
        self.decay_ratio = 3
        
        self.losses = []
        # 添加循环检查点
        self.iteration_checkpoints = []
        
        optimizer_D = Adam(lr=1e-4, beta_1=0.0, beta_2=0.9)
        optimizer_G = Adam(lr=1e-4, beta_1=0.0, beta_2=0.9)

        self.generator = Generator( (self.z_dim,) ,self.batch_size)
        self.discriminator = Discriminator(self.feature_dim, self.batch_size)
        
        self.generator.summary()
        self.discriminator.summary()
        
        self.generator.trainable = False
        real_img = Input(shape=self.img_shape)
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)
        fake = self.discriminator(fake_img)
        real = self.discriminator(real_img)

        self.discriminator_model = Model(inputs=[real_img, z_disc], outputs=[real, fake])
        self.discriminator_model.compile(loss=[self.eval_moments,self.eval_moments], optimizer=optimizer_D)
        self.discriminator.trainable = False
        self.generator.trainable = True
        
        z_gen = Input(shape=(self.z_dim,))
        img = self.generator(z_gen)
        fake_img = self.discriminator(img)
        self.generator_model = Model(z_gen, fake_img)
        self.generator_model.compile(loss=self.eval_moments, optimizer=optimizer_G)

    def train(self):
        batch_size = self.batch_size
        epochs = self.epochs
        (X_train, _), (_, _) = cifar10.load_data()
        X_train = (X_train.astype(np.float32)) / 127.5 - 1.0
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        for epoch in range(1,epochs+1):
            for _ in range(self.nb_learning):
                imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
                d_loss = self.discriminator_model.train_on_batch([imgs, noise], [negative_y, positive_y])

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            g_loss = self.generator_model.train_on_batch(noise, negative_y)        
            d_loss = d_loss[0] + d_loss[1]
            
            if (epoch + 1) % self.sample_interval == 0:
            
                # 输出训练过程数据
                print ("%d [D loss: %f] [G loss: %f]" % 
                         (epoch + 1, d_loss, g_loss))

                # 输出生成图像样本 
                self.sample_images(epoch + 1)
                
            if (epoch + 1) % (self.sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((d_loss, g_loss))
                
                self.iteration_checkpoints.append(epoch + 1)
        
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
        
        #保存训练后的生成图像
        z = np.random.normal(0, 1, (5000, self.z_dim))
        gen_imgs = self.generator.predict(z)
        
        return gen_imgs
    
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
        if not os.path.exists('cifar10_images-SphereGAN/'):
            os.makedirs('cifar10_images-SphereGAN/')
        fig.savefig("cifar10_images-SphereGAN/cifar10_%d.png" % iteration)
        plt.close()
            


# In[3]:


if __name__ == '__main__':
    SphereGAN = SphereGAN()
    gen = SphereGAN.train()


# In[5]:


if __name__ == '__main__':
    
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


# # IS

# In[6]:


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




