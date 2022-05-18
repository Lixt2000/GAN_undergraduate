#!/usr/bin/env python
# coding: utf-8

# # Conditional GAN

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from keras import backend as K

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


img_rows = 28
img_cols = 28
channels = 1
#输入图像维度
img_shape = (img_rows, img_cols, channels)

#噪音向量，用于输入至生成器
z_dim = 100

#数据集中类别的数量
num_classes = 10

adam_lr = 0.0002
adam_beta_1 = 0.5


# ## Generator

# In[3]:


def build_generator(img_shape, z_dim):

    model = Sequential()

    # 通过全连接层，将输入变为7x7x256的张量
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # 转置卷积层，张量从7x7x256变为14x14x128
    model.add(Conv2DTranspose(
                128, kernel_size=3, strides=2, padding='same'))

    # 批归一化
    model.add(BatchNormalization())

    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    
    # 转置卷积层，张量从14x14x128变为14x14x64
    model.add(Conv2DTranspose(
                64, kernel_size=3, strides=1, padding='same'))

    # 批归一化
    model.add(BatchNormalization())

    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    
    # 转置卷积层，张量从14x14x64变为28x28x1
    model.add(Conv2DTranspose(
                1, kernel_size=3, strides=2, padding='same'))

    # 带tanh激活函数的输出层
    model.add(Activation('tanh'))
    
    # 随机噪声z
    z = Input(shape=(z_dim,))
    
    # 条件标签:
    # G应该生成的指定数字，整数0～9
    label = Input(shape=(1,), dtype='int32')
    
    # 嵌入层:
    # 将标签转化为大小为z_dim的稠密向量
    # 生成形状为(batch_size, 1, z_dim)的三维张量
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)
    
    # 将嵌入的三维张量展平成形状为(batch_size, z_dim)的二维张量
    label_embedding = Flatten()(label_embedding)
    
    # 向量z和嵌入标签的元素级乘积
    joined_representation = multiply([z, label_embedding])
    
    #为给定的标签生成图像
    img = model(joined_representation)

    return Model([z, label], img)


# ## Discriminator

# In[4]:


def build_discriminator(img_shape):

    model = Sequential()
    
    # 卷积层, 张量从28x28x2变为14x14x64
    model.add(Conv2D(64, kernel_size=3, strides=2, 
                             input_shape=(img_rows, img_cols, 2), padding='same'))

    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))

    # 卷积层, 张量从14x14x64变为7x7x64
    model.add(Conv2D(64, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    
    # 批归一化
    model.add(BatchNormalization())
    
    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    
    # 卷积层, 张量从7x7x64变为3x3x128
    model.add(Conv2D(128, kernel_size=3, strides=2, 
                             input_shape=img_shape, padding='same'))
    
    # 批归一化
    model.add(BatchNormalization())
    
    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))

    # 展开张量，带sigmoid激活函数的输出层
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # 输入图像
    img = Input(shape=img_shape)
       
    # 条件标签:
    # G应该生成的指定数字，整数0～9
    label = Input(shape=(1,), dtype='int32')
    
    # 嵌入层:
    # 将标签转化为大小为z_dim的稠密向量
    # 生成形状为(batch_size, 1, 28*28*1)的三维张量
    label_embedding = Embedding(num_classes, np.prod(img_shape), input_length=1)(label)
    
    # 将嵌入的三维张量展平成形状为(batch_size, 28*28*1)的二维向量
    label_embedding = Flatten()(label_embedding)
    
    # 将嵌入标签调整为和输入图像一样的维度
    label_embedding = Reshape(img_shape)(label_embedding)
    
    # 将图像与其嵌入标签连接
    concatenated = Concatenate(axis=-1)([img, label_embedding])
    
    #对“图像-标签”对进行分类
    prediction = model(concatenated)
    
    return Model([img, label], prediction)


# ## Build the Model

# In[5]:


# -------------------------
#  建立鉴别器
# -------------------------

# 构建并编译鉴别器 
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), metrics=['accuracy'])

# ---------------------
#  建立生成器
# ---------------------

generator = build_generator(img_shape, z_dim)

# 噪声和标签类别作为生成器的输入
# 为指定标签生成图像
z = Input(shape=(z_dim,))
label = Input(shape=(1,))
img = generator([z, label])

# 生成器训练时，鉴别器参数保持不变
discriminator.trainable = False

prediction = discriminator([img, label])

# 构建并编译鉴别器固定的CGAN模型来训练生成器（鉴别器权重固定）
cgan = Model([z, label], prediction)
cgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1))


# In[6]:


accuracies = []
losses = []
# 添加循环检查点
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    
    # 载入数据集
    (X_train, y_train), (_, _) = mnist.load_data()

    # 缩放范围到(-1, 1)
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # 真实图像标签标记为1，虚假图像标签标记为0
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for iteration in range(iterations):

        # ---------------------
        #  训练鉴别器
        # ---------------------

        # 生成一批量伪样本及其标签
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # 生成一批伪图像
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict([z, labels])

        # 训练鉴别器
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  训练生成器
        # ---------------------
        
        # 生成一批伪图像
        z = np.random.normal(0, 1, (batch_size, z_dim))
        
        # 生成一批随机标签
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        # 训练生成器
        g_loss = cgan.train_on_batch([z, labels], real)
        
        if (iteration + 1) % sample_interval == 0:

            # 输出训练过程数据
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))

            # 输出生成图像样本 
            sample_images(generator, iteration + 1)
            
        if (iteration + 1) % (sample_interval*10) == 0:
                
            # 保存损失和准确度
            losses.append((d_loss[0], g_loss))
            accuracies.append(100*d_loss[1])
            iteration_checkpoints.append(iteration + 1)
    
    #保存训练后的生成图像
    z = np.random.normal(0, 1, (5000, z_dim))
    gen_imgs = generator.predict([z, labels])
        
    return gen_imgs 


# In[7]:


def sample_images(generator, iteration):
    
    image_grid_rows=2
    image_grid_columns=5

    # 随机噪声采样
    z = np.random.normal(0, 1, 
              (image_grid_rows * image_grid_columns, z_dim))
    
    labels = np.arange(0, 10).reshape(-1, 1)
    
    # 从随机噪声生成图像
    gen_imgs = generator.predict([z, labels])

    # 图像像素值缩放到[0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
 
    # 设置图像网格
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                    figsize=(10,4), sharey=True, sharex=True)
    
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 输出图像网格
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            axs[i,j].set_title("Digit: %d" % labels[cnt])
            cnt += 1
            
    # 保存图像
    if not os.path.exists('mnist_images-CGAN/'):
        os.makedirs('mnist_images-CGAN/')
    fig.savefig("mnist_images-CGAN/mnist_%d.png" % iteration)
    plt.close()


# In[8]:


# Suppress warnings because the warning Keras gives us about non-trainable parameters is by design:# Suppr 
# The Generator trainable parameters are intentionally held constant during Discriminator training and vice versa
import warnings; warnings.simplefilter('ignore')


# In[9]:


iterations = 30000
batch_size = 64
sample_interval = 100

# 按照指定的迭代次数训练SGAN
gen_imgs = train(iterations, batch_size, sample_interval)


# In[11]:


(X_train, y_train), (_, _) = mnist.load_data()

idx = np.random.randint(0, X_train.shape[0], 5000)
labels = y_train[idx]
z = np.random.normal(0, 1, (5000, z_dim))
gen_imgs = generator.predict([z, labels])


# In[12]:


image_grid_rows = 10
image_grid_columns = 5

# 随机噪声样本
z = np.random.normal(0, 1, 
          (image_grid_rows * image_grid_columns, z_dim))

labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
labels_to_generate = labels_to_generate.flatten().reshape(-1, 1)

# 根据随机噪声生成图像
gen_imgs_trained = generator.predict([z, labels_to_generate])

# 缩放范围到[0, 1]
gen_imgs_trained = 0.5 * gen_imgs_trained + 0.5

# 绘图设置
fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                figsize=(10, 20), sharey=True, sharex=True)

cnt = 0
for i in range(image_grid_rows):
    for j in range(image_grid_columns):
        # 输出图像网格
        axs[i,j].imshow(gen_imgs_trained[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        axs[i,j].set_title("Digit: %d" % labels_to_generate[cnt]) ## NEW
        cnt += 1

# 保存图像
plt.savefig("mnist_images/generated image with labels.png")


# In[13]:


losses = np.array(losses)

# 绘制鉴别器和生成器的损失图像
plt.figure(figsize=(10,5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator Loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator Loss")
plt.xticks(iteration_checkpoints, rotation=90)
plt.title("Training Losses")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()


# In[14]:


accuracies = np.array(accuracies)

# 绘制鉴别器的准确度图像
plt.figure(figsize=(10,5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator Accuracy")
plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))
plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()


# # IS

# In[15]:


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
(images1, _), (_, _) = mnist.load_data()
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


# ----
