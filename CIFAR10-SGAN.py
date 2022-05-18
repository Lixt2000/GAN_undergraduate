#!/usr/bin/env python
# coding: utf-8

# # Semi-Supervised GAN

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from keras import backend as K
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from keras.utils import plot_model, np_utils
from keras import initializers

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ## Dataset

# In[54]:


# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 输出部分真实图像
fig = plt.figure(figsize=(8,3))
for i in range(0, 10):
    plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
    plt.imshow(X_train[i])
    
plt.tight_layout()

# 输出带有标签的部分真实图像
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

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

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


# In[55]:


class Dataset:
    
    def __init__(self, num_labeled):
        
        # 训练中使用的有标签图像的数量
        self.num_labeled = num_labeled       
        
        def preprocess_imgs(x):
            # 灰度像素值(0, 255)缩放范围到(-1, 1)
            x = (x.astype(np.float32) - 127.5) / 127.5
            # 将图像尺寸扩展到宽 x 高 x 通道数
            x = np.expand_dims(x, axis=3)
            return x

        def preprocess_labels(y):
            return y.reshape(-1, 1)
     
        # 训练
        #self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(Y_train)
        
        # 测试
        #self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(Y_test)

    def batch_labeled(self, batch_size):
        # 获取随机批量的有标签图像及其标签
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = X_train[idx]
        labels = y_train[idx]
        return imgs, labels
    
    def batch_unlabeled(self, batch_size):
        # 获取随机批量的无标签图像
        idx = np.random.randint(self.num_labeled, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        return imgs
    
    def training_set(self):
        #X_train = X_train[range(self.num_labeled)]
        Y_train = self.y_train[range(self.num_labeled)]
        #return X_train, Y_train
        return Y_train
    
    def test_set(self):
        #return X_test, self.y_test
        return self.y_test


# In[56]:


num_labeled = 100

dataset = Dataset(num_labeled)


# # Semi-Supervied GAN

# In[5]:


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


# ## Generator

# In[6]:


init = initializers.RandomNormal(stddev=0.02)

# Generator network
generator = Sequential()

# FC: 2x2x512
generator.add(Dense(2*2*512, input_shape=(z_dim,), kernel_initializer=init))
generator.add(Reshape((2, 2, 512)))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# # Conv 1: 4x4x256
generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 2: 8x8x128
generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 3: 16x16x64
generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 4: 32x32x3
generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh'))


# ## Discriminator

# In[7]:


#核心鉴别器

# imagem shape 32x32x3
img_shape = X_train[0].shape

# Discriminator network
discriminator = Sequential()

# Conv 1: 16x16x64
discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                         input_shape=(img_shape), kernel_initializer=init))
discriminator.add(LeakyReLU(0.2))

# Conv 2:
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3: 
discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3: 
discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# FC
discriminator.add(Flatten())

# Output
discriminator.add(Dense(num_classes))


# In[8]:


#鉴别器有监督部分
discriminator_supervised = Sequential()

discriminator_supervised.add(discriminator)

# Softmax激活函数，输出真实类别的预测概率分布
discriminator_supervised.add(Activation('softmax'))


# In[9]:


#鉴别器无监督部分
discriminator_unsupervised = Sequential()

discriminator_unsupervised.add(discriminator)

def predict(x):
    # 将真实类别的分布转换为二元真-假概率
    prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
    return prediction

discriminator_unsupervised.add(Lambda(predict))


# ## Build the Model

# In[10]:


# -------------------------
#  建立鉴别器模型
# -------------------------

# 构建并编译有监督训练鉴别器
discriminator_supervised.compile(
    loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1))

# 构建并编译无监督训练鉴别器
discriminator_unsupervised.compile(loss='binary_crossentropy', optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1))

# ---------------------
#  建立生成器模型
# ---------------------

# 生成器训练时，鉴别器参数保持不变
discriminator_unsupervised.trainable = False

z = Input(shape=(z_dim,))
img = generator(z)
prediction = discriminator_unsupervised(img)
sgan = Model(inputs=z, outputs=prediction)
sgan.compile(Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy',
            metrics=['binary_accuracy'])


# ## Training

# In[11]:


d_accuracies = []
d_losses = []
g_losses = []
# 添加循环检查点
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    
    # 真实图像标签标记为1，虚假图像标签标记为0
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        
        # -------------------------
        #  训练鉴别器
        # -------------------------
        
        # 获取有标签样本
        imgs, labels = dataset.batch_labeled(batch_size)
        
        # 独热编码标签
        labels = to_categorical(labels, num_classes=num_classes)

        # 获取无标签样本
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        # 生成一批伪图像
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        # 训练有标签的真实样本
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)
        
        # 训练无标签的真实样本
        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)
           
        # 训练伪样本
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
        
        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  训练生成器
        # ---------------------
        
        # 生成一批伪图像
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # 训练生成器
        g_loss = sgan.train_on_batch(z, np.ones((batch_size, 1)))
        
        if (iteration + 1) % sample_interval == 0:

            # 输出训练过程数据
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration + 1, d_loss_supervised, 100 * accuracy, g_loss[-1]))

            # 输出生成图像样本 
            sample_images(generator, iteration + 1)
            
        if (iteration + 1) % (sample_interval*10) == 0:
                
            # 保存损失和准确度
            g_losses.append(g_loss[-1])
            d_losses.append(d_loss_supervised)
            d_accuracies.append(accuracy)
            iteration_checkpoints.append(iteration + 1)
            
    #保存训练后的生成图像
    z = np.random.normal(0, 1, (5000, z_dim))
    gen_imgs = generator.predict(z)
        
    return gen_imgs


# In[12]:


def sample_images(generator, iteration):

    image_grid_rows=4
    image_grid_columns=4
    
    # 随机噪声样本
    z = np.random.normal(0, 1, 
              (image_grid_rows * image_grid_columns, z_dim))

    # 根据随机噪声生成图像 
    gen_imgs = generator.predict(z)

    # 缩放范围到[0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    # 绘图设置
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, 
                                    figsize=(4,4), sharey=True, sharex=True)
    
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output image grid
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
            
    # 保存图像
    if not os.path.exists('cifar10_images-SGAN/'):
        os.makedirs('cifar10_images-SGAN/')
    fig.savefig("cifar10_images-SGAN/cifar10_%d.png" % iteration)
    plt.close()


# ## Train the Model and Inspect Output

# In[13]:


# Suppress warnings because the warning Keras gives us about non-trainable parameters is by design:
# The Generator trainable parameters are intentionally held constant during Discriminator training and vice versa
import warnings; warnings.simplefilter('ignore')


# In[14]:


iterations = 30000
batch_size = 64
sample_interval = 100

# 按照指定的迭代次数训练SGAN
gen_imgs = train(iterations, batch_size, sample_interval)


# In[15]:


# 绘制鉴别器有监督分类损失和生成器损失
plt.figure(figsize=(10,5))
plt.plot(np.array(d_losses), label="Discriminator – Supervised Loss")
plt.plot(np.array(g_losses), label="Generator Loss")
plt.title("Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()


# In[16]:


# 绘制鉴别器有监督分类精度
plt.figure(figsize=(10,5))
plt.plot(np.array(d_accuracies), label="Accuracy")
plt.title("Discriminator – Supervised Classification Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy(100%)")
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

# In[18]:


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
images1 = np.clip(images1, -1, 1)
images2 = np.clip(images2, -1, 1)
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


# ## SGAN Classifier – Training and Test Accuracy 

# In[67]:


x = X_train[range(num_labeled)]
y = dataset.training_set()
y = to_categorical(y, num_classes=num_classes)

# 在训练集上计算分类准确率
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Training Accuracy: %.2f%%" % (100 * accuracy))


# In[68]:


x = X_test
y = dataset.test_set()
y = y[range(10000)]
y = to_categorical(y, num_classes=num_classes)

# 在测试集上计算分类准确率
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# ---

# # Fully-Supervised Classifier

# In[61]:


# 建立有着与SGAN鉴别器相同网络结构的全监督分类器
mnist_classifier = discriminator_supervised
mnist_classifier.compile(
            loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())


# In[62]:


imgs = X_train[range(num_labeled)]
labels = dataset.training_set()

# 独热编码标签
labels = to_categorical(labels, num_classes=num_classes)

# 训练分类器
training = cifar10_classifier.fit(x=imgs, y=labels, batch_size=32, epochs=50, verbose=1)
losses = training.history['loss']
accuracies = training.history['accuracy']


# In[63]:


# 绘制分类损失
plt.figure(figsize=(10,5))
plt.plot(np.array(losses), label="Loss")
plt.title("Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# In[64]:


# 绘制分类准确率
plt.figure(figsize=(10,5))
plt.plot(np.array(accuracies), label="Accuracy")
plt.title("Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy（100%）")
plt.legend()


# In[65]:


x = X_train[range(num_labeled)]
y = dataset.training_set()
y = to_categorical(y, num_classes=num_classes)

# 在训练集上计算分类准确率
_, accuracy = cifar10_classifier.evaluate(x, y)
print("Training Accuracy: %.2f%%" % (100 * accuracy))


# In[66]:


x = X_test
y = dataset.test_set()
y = y[range(10000)]
y = to_categorical(y, num_classes=num_classes)

# 在测试集上计算分类准确率
_, accuracy = cifar10_classifier.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))


# ---
