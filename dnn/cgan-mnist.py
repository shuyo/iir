#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MNIST generator based on Conditional Generative Adversarial Networks with Tensorflow
# (M. Mirza and S. Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014.)

# This code is available under the MIT License.
# (c)2016 Nakatani Shuyo / Cybozu Labs Inc.

import numpy, math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# model parameter
noise_dim = 10 # input noise size of Generator
Dhidden = 256  # hidden units of Discriminator's network
Ghidden = 512  # hidden units of Generator's network
K = 8          # maxout units of Discriminator

mini_batch_size = 50
epoch = 50
nsamples = 7 # drawing samples

mnist = input_data.read_data_sets("data/", one_hot=True)
N, num_features = mnist.train.images.shape
_, num_labels = mnist.train.labels.shape
period = N // mini_batch_size

X = tf.placeholder(tf.float32, shape=(None, num_features))
Y = tf.placeholder(tf.float32, shape=(None, num_labels))
Z = tf.placeholder(tf.float32, shape=(None, noise_dim))
keep_prob = tf.placeholder(tf.float32)

GW1z = tf.Variable(tf.random_normal([noise_dim, Ghidden], stddev=0.1), name="GW1z")
GW1y = tf.Variable(tf.random_normal([num_labels, Ghidden], stddev=0.1), name="GW1y")
Gb1 = tf.Variable(tf.zeros(Ghidden), name="Gb1")
GW2 = tf.Variable(tf.random_normal([Ghidden, num_features], stddev=0.1), name="GW2")
Gb2 = tf.Variable(tf.zeros(num_features), name="Gb2")

DW1x = tf.Variable(tf.random_normal([num_features, K * Dhidden], stddev=0.01), name="DW1x")
DW1y = tf.Variable(tf.random_normal([num_labels, K * Dhidden], stddev=0.01), name="DW1y")
Db1 = tf.Variable(tf.zeros(K * Dhidden), name="Db1")
DW2 = tf.Variable(tf.random_normal([Dhidden, 1], stddev=0.01), name="DW2")
Db2 = tf.Variable(tf.zeros(1), name="Db2")

def discriminator(x, y):
    u = tf.reshape(tf.matmul(x, DW1x) + tf.matmul(y, DW1y) + Db1, [-1, K, Dhidden])
    Dh1 = tf.nn.dropout(tf.reduce_max(u, reduction_indices=[1]), keep_prob)
    return tf.nn.sigmoid(tf.matmul(Dh1, DW2) + Db2)

Gh1 = tf.nn.relu(tf.matmul(Z, GW1z) + tf.matmul(Y, GW1y) + Gb1)
G = tf.nn.sigmoid(tf.matmul(Gh1, GW2) + Gb2)
DG = discriminator(G, Y)

Dloss = -tf.reduce_mean(tf.log(discriminator(X, Y)) + tf.log(1 - DG))
Gloss = tf.reduce_mean(tf.log(1 - DG) - tf.log(DG + 1e-9)) # the second term for stable learning

vars = tf.trainable_variables()
Dvars = [v for v in vars if v.name.startswith("D")]
Gvars = [v for v in vars if v.name.startswith("G")]

Doptimizer = tf.train.AdamOptimizer().minimize(Dloss, var_list=Dvars)
Goptimizer = tf.train.AdamOptimizer().minimize(Gloss, var_list=Gvars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for e in range(epoch):
    dloss = gloss = 0.0
    for i in range(period):
        x, y = mnist.train.next_batch(mini_batch_size)
        z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
        loss, _ = sess.run([Dloss, Doptimizer], feed_dict={X:x, Y:y, Z:z, keep_prob:0.5})
        dloss += loss
        z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
        loss, _ = sess.run([Gloss, Goptimizer], feed_dict={Y:y, Z:z, keep_prob:1.0})
        gloss += loss

    print("%d: dloss=%.5f, gloss=%.5f" % (e+1, dloss / period, gloss / period))
    if math.isnan(dloss) or math.isnan(gloss):
        sess.run(tf.initialize_all_variables()) # initialize & retry if NaN

def save_figures(path, z):
    fig = plt.figure()
    fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    for i in range(num_labels):
        y = numpy.zeros((z.shape[0], num_labels))
        y[:,i] = 1
        Gz = sess.run(G, feed_dict={Y:y, Z: z})
        for j in range(nsamples):
            ax = fig.add_subplot(nsamples, num_labels, j * num_labels + i + 1)
            ax.axis("off")
            ax.imshow(Gz[j,:].reshape((28,28)), cmap=plt.get_cmap("gray"))
    fig.savefig(path)
    plt.close(fig)

z = numpy.random.uniform(-1, 1, size=(nsamples, noise_dim))
#z[:,0] = numpy.arange(0, nsamples) / (nsamples - 1) * 2 - 1
save_figures("cgan-mnist.png", z)

