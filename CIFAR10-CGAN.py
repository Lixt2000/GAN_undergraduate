#!/usr/bin/env python
# coding: utf-8

# # Conditional GAN

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from keras import backend as K

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


img_rows = 32
img_cols = 32
channels = 3
#输入图像维度
img_shape = (img_rows, img_cols, channels)

#噪音向量，用于输入至生成器
z_dim = 100

#数据集中类别的数量
num_classes = 10

adam_lr = 0.0002
adam_beta_1 = 0.5


# In[3]:


# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[4]:


num_classes = len(np.unique(y_train))
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = X_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    img = features_idx[img_num,::]
    ax.set_title(class_names[i])
    plt.imshow(img)
    
plt.tight_layout()


# In[5]:


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[6]:


if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    input_shape = (3, 32, 32)
else:
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)
    
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# the generator is using tanh activation, for which we need to preprocess 
# the image data into the range between -1 and 1.

X_train = np.float32(X_train)
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)

X_test = np.float32(X_test)
X_test = (X_test / 255 - 0.5) * 2
X_test = np.clip(X_test, -1, 1)

print('X_train reshape:', X_train.shape)
print('X_test reshape:', X_test.shape)


# ## Generator

# In[7]:


# latent space dimension
z = Input(shape=(100,))

# classes
labels = Input(shape=(10,))

# Generator network
merged_layer = Concatenate()([z, labels])

# FC: 2x2x512
generator = Dense(2*2*512, activation='relu')(merged_layer)
generator = BatchNormalization(momentum=0.9)(generator)
generator = LeakyReLU(alpha=0.1)(generator)
generator = Reshape((2, 2, 512))(generator)

# # Conv 1: 4x4x256
generator = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(generator)
generator = BatchNormalization(momentum=0.9)(generator)
generator = LeakyReLU(alpha=0.1)(generator)

# Conv 2: 8x8x128
generator = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(generator)
generator = BatchNormalization(momentum=0.9)(generator)
generator = LeakyReLU(alpha=0.1)(generator)

# Conv 3: 16x16x64
generator = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(generator)
generator = BatchNormalization(momentum=0.9)(generator)
generator = LeakyReLU(alpha=0.1)(generator)

# Conv 4: 32x32x3
generator = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(generator)

# generator = Model(inputs=[z, labels], outputs=out_g)
generator = Model(inputs=[z, labels], outputs=generator, name='generator')


# ## Discriminator

# In[8]:


# input image
img_input = Input(shape=(X_train[0].shape))

# Conv 1: 16x16x64
discriminator = Conv2D(64, kernel_size=5, strides=2, padding='same')(img_input)
discriminator = BatchNormalization(momentum=0.9)(discriminator)
discriminator = LeakyReLU(alpha=0.1)(discriminator)

# Conv 2:
discriminator = Conv2D(128, kernel_size=5, strides=2, padding='same')(discriminator)
discriminator = BatchNormalization(momentum=0.9)(discriminator)
discriminator = LeakyReLU(alpha=0.1)(discriminator)

# Conv 3: 
discriminator = Conv2D(256, kernel_size=5, strides=2, padding='same')(discriminator)
discriminator = BatchNormalization(momentum=0.9)(discriminator)
discriminator = LeakyReLU(alpha=0.1)(discriminator)

# Conv 4: 
discriminator = Conv2D(512, kernel_size=5, strides=2, padding='same')(discriminator)
discriminator = BatchNormalization(momentum=0.9)(discriminator)
discriminator = LeakyReLU(alpha=0.1)(discriminator)

# FC
discriminator = Flatten()(discriminator)

# Concatenate 
merged_layer = Concatenate()([discriminator, labels])
discriminator = Dense(512, activation='relu')(merged_layer)
    
# Output
discriminator = Dense(1, activation='sigmoid')(discriminator)

discriminator = Model(inputs=[img_input, labels], outputs=discriminator, name='discriminator')


# ## Build the Model

# In[9]:


# -------------------------
#  建立鉴别器
# -------------------------

# 构建并编译鉴别器 
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), metrics=['binary_accuracy'])

# ---------------------
#  建立生成器
# ---------------------

# 生成器训练时，鉴别器参数保持不变
discriminator.trainable = False

# 噪声和标签类别作为生成器的输入
# 为指定标签生成图像
label = Input(shape=(10,), name='label')
z = Input(shape=(100,), name='z')

fake_img = generator([z, label])
validity = discriminator([fake_img, label])

# 构建并编译鉴别器固定的CGAN模型来训练生成器（鉴别器权重固定）
cgan = Model([z, label], validity, name='adversarial')
cgan.compile(Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy',
            metrics=['binary_accuracy'])


# In[10]:


accuracies = []
losses = []
# 添加循环检查点
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):

    # 真实图像标签标记为1，虚假图像标签标记为0
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for iteration in range(iterations):

        # ---------------------
        #  训练鉴别器
        # ---------------------

        # 生成一批量伪样本及其标签
        z = np.random.normal(loc=0, scale=1, size=(batch_size, z_dim))
        random_labels = to_categorical(np.random.randint(0, 10, batch_size).reshape(-1, 1), num_classes=10)
        # 生成一批伪图像
        gen_imgs = generator.predict_on_batch([z, random_labels])
        
        #获取真实图像
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        real_labels = to_categorical(y_train[idx].reshape(-1, 1), num_classes=10)
        
        # 训练鉴别器
        d_loss_real = discriminator.train_on_batch([imgs, real_labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, random_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  训练生成器
        # ---------------------
        
        # 生成一批伪图像
        z = np.random.normal(loc=0, scale=1, size=(batch_size, z_dim))
        # 生成一批随机标签
        random_labels = to_categorical(np.random.randint(0, 10, batch_size).reshape(-1, 1), num_classes=10)

        # 训练生成器
        g_loss = cgan.train_on_batch([z, random_labels], real)
        
        if (iteration + 1) % sample_interval == 0:

            # 输出训练过程数据
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss[-1]))

            # 输出生成图像样本 
            sample_images(generator, iteration + 1)
            
        if (iteration + 1) % (sample_interval*10) == 0:
                
            # 保存损失和准确度
            losses.append((d_loss[0], g_loss[-1]))
            accuracies.append(100*d_loss[1])
            iteration_checkpoints.append(iteration + 1)
    
    #保存训练后的生成图像
    z = np.random.normal(0, 1, (5000, z_dim))
    labels = to_categorical(np.random.randint(0, 10, 5000).reshape(-1, 1), num_classes=10)
    gen_imgs = generator.predict([z, labels])
        
    return gen_imgs 


# In[11]:


def sample_images(generator, iteration):
    
    image_grid_rows=2
    image_grid_columns=5
    samples=image_grid_rows * image_grid_columns

    # 随机噪声采样
    z = np.random.normal(0, 1, 
              (image_grid_rows * image_grid_columns, z_dim))
    
    labels = to_categorical(np.arange(0, 10).reshape(-1, 1), num_classes=10)
    
    x_fake = generator.predict([z, labels])
    x_fake = np.clip(x_fake, -1, 1)
    x_fake = (x_fake + 1) * 127
    x_fake = np.round(x_fake).astype('uint8')

    for k in range(samples):
        plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
        plt.imshow(x_fake[k])
        plt.title(class_names[k])
        
    plt.tight_layout()
            
    # 保存图像
    if not os.path.exists('cifar10_images-CGAN/'):
        os.makedirs('cifar10_images-CGAN/')
    fig.savefig("cifar10_images-CGAN/cifar10_%d.png" % iteration)
    plt.close()


# In[12]:


# Suppress warnings because the warning Keras gives us about non-trainable parameters is by design:# Suppr 
# The Generator trainable parameters are intentionally held constant during Discriminator training and vice versa
import warnings; warnings.simplefilter('ignore')


# In[13]:


iterations = 30000
batch_size = 64
sample_interval = 100

# 按照指定的迭代次数训练SGAN
gen_imgs = train(iterations, batch_size, sample_interval)


# In[14]:


image_grid_rows = 10
image_grid_columns = 5
samples =image_grid_rows * image_grid_columns

# 随机噪声样本
z = np.random.normal(0, 1, 
          (image_grid_rows * image_grid_columns, z_dim))

labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
labels = to_categorical(labels_to_generate.flatten().reshape(-1, 1), num_classes=10)
    
x_fake = generator.predict([z, labels])
x_fake = np.clip(x_fake, -1, 1)
x_fake = (x_fake + 1) * 127
x_fake = np.round(x_fake).astype('uint8')

# 绘图设置
fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                figsize=(10, 20), sharey=True, sharex=True)

for i in range(image_grid_rows):
    for j in range(image_grid_columns):
        # 输出图像网格
        axs[i,j].imshow(x_fake[5 * i + j])
        axs[i,j].axis('off')
        axs[i,j].set_title(class_names[i])

# 保存图像
plt.savefig("cifar10_images/generated image with labels.png")


# In[15]:


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


# In[16]:


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

# In[17]:


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


# ----
