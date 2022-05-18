#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


class RandomWeightedAverage(_Merge):
    """给真实样本和生成样本提供一个随机权重平均"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# In[7]:


class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # 参照文献中的参数与优化器设置
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)
        
        self.losses = []
        self.accuracies = []
        # 添加循环检查点
        self.iteration_checkpoints = []

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
        z_disc = Input(shape=(self.latent_dim,))
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
        z_gen = Input(shape=(100,))
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

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
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

        noise = Input(shape=(self.latent_dim,))
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

    def train(self, epochs, batch_size, sample_interval=50):

        # 加载数据
        (X_train, _), (_, _) = mnist.load_data()

        # 范围为(-1, 1)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  训练评论器
                # ---------------------

                # 抽取随机真实样本
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # 抽取噪声，输入生成器
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # 训练评论器
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  训练生成器
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # 如果是保存间隔 => 保存生成的图像样本
            if (epoch + 1) % sample_interval == 0:
                # 输出过程
                print ("%d [D loss: %f] [G loss: %f]" % (epoch + 1, d_loss[0], g_loss))
                
                self.sample_images(epoch + 1)
                
            if (epoch + 1) % (sample_interval*10) == 0:
                
                # 保存损失和准确度
                self.losses.append((1 - d_loss[0], g_loss))
                self.accuracies.append(100*d_loss[1])
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

    def sample_images(self, epoch):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 重构范围到[0,1]
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        if not os.path.exists('mnist_images-WGAN-GP/'):
            os.makedirs('mnist_images-WGAN-GP/')
        fig.savefig("mnist_images-WGAN-GP/mnist_%d.png" % epoch)
        plt.close()


# In[8]:


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)


# In[ ]:




