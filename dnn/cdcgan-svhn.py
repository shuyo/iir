#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SVHN generator based on DCGAN and Conditional GAN with Tensorflow
# Radford, A., Metz, L., and Chintala, S. Unsupervised representation learning with deep convolutional generative adversarial networks. 2016.
# M. Mirza and S. Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014.

# This code is available under the MIT License.
# (c)2016 Nakatani Shuyo / Cybozu Labs Inc.

import numpy, math, time
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt

# model parameter
noise_dim = 100 # input noise size of Generator
Dhidden = [64, 128, 256]    # hidden units of Discriminator's network
Ghidden = [512, 256, 128]  # hidden units of Generator's network

mini_batch_size = 128
samples=(8,10)  # samples drawing size
nsamples = samples[0] * samples[1]
assert nsamples <= mini_batch_size
epoch = 200

# download train_32x32.mat in advance from http://ufldl.stanford.edu/housenumbers/
svhn = scipy.io.loadmat("data/train_32x32.mat")
train_data = svhn["X"]
train_labels = svhn["y"].flatten()
#train_data = train_data[:, :, :, :256] # small dataset
#train_labels = train_labels[:256]
num_labels = 10
train_labels[train_labels==10] = 0

fig_width, fig_height, n_channels, N = train_data.shape
train_data = train_data.reshape(fig_width * fig_height * n_channels, N)
train_data -= train_data.min(axis=0)
train_data = (numpy.array(train_data, dtype=numpy.float32) / train_data.max(axis=0)).T.reshape(N, fig_width, fig_height, n_channels)
period = N // mini_batch_size

X = tf.placeholder(tf.float32, shape=(None, fig_width, fig_height, n_channels))
Y = tf.placeholder(tf.float32, shape=(None, num_labels))
Z = tf.placeholder(tf.float32, shape=(None, noise_dim))
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("G"):
    GW0 = tf.Variable(tf.random_normal([noise_dim, Ghidden[0]*4*4], stddev=0.01))
    GW0y = tf.Variable(tf.random_normal([num_labels, Ghidden[0]*4*4], stddev=0.01))
    Gb0 = tf.Variable(tf.zeros(Ghidden[0]))
    GW1 = tf.Variable(tf.random_normal([5, 5, Ghidden[1], Ghidden[0]], stddev=0.01))
    Gb1 = tf.Variable(tf.zeros(Ghidden[1]))
    GW2 = tf.Variable(tf.random_normal([5, 5, Ghidden[2], Ghidden[1]], stddev=0.01))
    Gb2 = tf.Variable(tf.zeros(Ghidden[2]))
    GW3 = tf.Variable(tf.random_normal([5, 5, n_channels, Ghidden[2]], stddev=0.01))
    Gb3 = tf.Variable(tf.zeros(n_channels))

# batch normalization & relu
def bn(u):
    mean, variance = tf.nn.moments(u, axes=[0, 1, 2])
    return tf.nn.relu(tf.nn.batch_normalization(u, mean, variance, None, None, 1e-5))

Gh0 = bn(tf.nn.bias_add(tf.reshape(tf.matmul(Z, GW0)+tf.matmul(Y,GW0y), [-1, fig_width//8, fig_height//8, Ghidden[0]]), Gb0))
Gh1 = bn(tf.nn.bias_add(tf.nn.conv2d_transpose(Gh0, GW1, [mini_batch_size, fig_width//4, fig_height//4, Ghidden[1]], [1, 2, 2, 1]), Gb1))
Gh2 = bn(tf.nn.bias_add(tf.nn.conv2d_transpose(Gh1, GW2, [mini_batch_size, fig_width//2, fig_height//2, Ghidden[2]], [1, 2, 2, 1]), Gb2))
G = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d_transpose(Gh2, GW3, [mini_batch_size, fig_width, fig_height, n_channels], [1, 2, 2, 1]), Gb3))

with tf.variable_scope("D"):
    DW0 = tf.Variable(tf.random_normal([5, 5, n_channels, Dhidden[0]], stddev=0.01))
    DW0y = tf.Variable(tf.random_normal([num_labels, (fig_width//2)*(fig_height//2)*Dhidden[0]], stddev=0.01))
    Db0 = tf.Variable(tf.zeros(Dhidden[0]))
    DW1 = tf.Variable(tf.random_normal([5, 5, Dhidden[0], Dhidden[1]], stddev=0.01))
    Db1 = tf.Variable(tf.zeros(Dhidden[1]))
    DW2 = tf.Variable(tf.random_normal([5, 5, Dhidden[1], Dhidden[2]], stddev=0.01))
    Db2 = tf.Variable(tf.zeros(Dhidden[2]))
    DW3 = tf.Variable(tf.random_normal([(fig_width//8)*(fig_height//8)*Dhidden[2], 1], stddev=0.01))
    Db3 = tf.Variable(tf.zeros(1))

# batch normalization & leaky relu
def bnl(u, a=0.2):
    mean, variance = tf.nn.moments(u, axes=[0, 1, 2])
    b = tf.nn.batch_normalization(u, mean, variance, None, None, 1e-5)
    return tf.maximum(a * b, b)

def discriminator(xx):
    Dh0 = bnl(tf.nn.bias_add(tf.nn.conv2d(xx, DW0, [1, 2, 2, 1], padding='SAME')+tf.reshape(tf.matmul(Y,DW0y),[-1,(fig_width//2),(fig_height//2),Dhidden[0]]), Db0))
    Dh1 = bnl(tf.nn.bias_add(tf.nn.conv2d(Dh0, DW1, [1, 2, 2, 1], padding='SAME'), Db1))
    Dh2 = bnl(tf.nn.bias_add(tf.nn.conv2d(Dh1, DW2, [1, 2, 2, 1], padding='SAME'), Db2))
    return tf.nn.sigmoid(tf.matmul(tf.reshape(Dh2, [-1, (fig_width//8)*(fig_height//8)*Dhidden[2]]), DW3) + Db3)

DG = discriminator(G)
Dloss = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - DG))
Gloss = tf.reduce_mean(tf.log(1 - DG) - tf.log(DG + 1e-9)) # the second term for stable learning

vars = tf.trainable_variables()
Dvars = [v for v in vars if v.name.startswith("D")]
Gvars = [v for v in vars if v.name.startswith("G")]

Doptimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(Dloss, var_list=Dvars)
Goptimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(Gloss, var_list=Gvars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

def save_figure(path, z, y):
    Gz = sess.run(G, feed_dict={Z: z, Y: y})
    #plt.ion()
    fig = plt.gcf()
    fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    for i in range(nsamples):
        ax = fig.add_subplot(samples[0], samples[1], i + 1)
        ax.axis("off")
        ax.imshow(Gz[i,:,:,:])
    plt.savefig(path)
    plt.draw()
    plt.pause(0.01)

t0 = time.time()
drawz = numpy.random.uniform(-1, 1, size=(mini_batch_size//num_labels+1, noise_dim)).repeat(num_labels, axis=0)[:mini_batch_size]
drawy = numpy.tile(numpy.eye(num_labels),(mini_batch_size//num_labels+1,1))[:mini_batch_size]

for e in range(epoch):
    index = numpy.random.permutation(N)
    dloss = gloss = 0.0
    for i in range(period):
        idx = index[i*mini_batch_size:(i+1)*mini_batch_size]
        x = train_data[idx, :]
        y = numpy.zeros((mini_batch_size, num_labels), dtype=numpy.float32)
        y[numpy.arange(mini_batch_size), train_labels[idx]] = 1
        z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
        loss, _ = sess.run([Dloss, Doptimizer], feed_dict={X:x, Y:y, Z:z, keep_prob:0.5})
        dloss += loss

        y = numpy.zeros((mini_batch_size, num_labels), dtype=numpy.float32)
        y[numpy.arange(mini_batch_size), numpy.random.randint(0,10,mini_batch_size)] = 1
        z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
        loss, _ = sess.run([Gloss, Goptimizer], feed_dict={Y:y, Z:z, keep_prob:1.0})
        gloss += loss

        if math.isnan(dloss) or math.isnan(gloss):
            sess.run(tf.initialize_all_variables()) # initialize & retry if NaN
            print("...initialize parameters for nan...")
            dloss = gloss = 0.0

    print("%d: dloss=%.5f, gloss=%.5f, time=%.1f" % (e+1, dloss / period, gloss / period, time.time()-t0))
    save_figure("png/cdcgan-svhn-%03d.png" % (e+1), drawz, drawy)

saver = tf.train.Saver()
saver.save(sess, "cdcgan-svhn.model")
