#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MNIST generator based on Generative Adversarial Networks with Tensorflow
# (I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, pages 2672–2680. 2014.)

# This code is available under the MIT License.
# (c)2016 Nakatani Shuyo / Cybozu Labs Inc.

import numpy, math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# model parameter
noise_dim = 32 # input noise size of Generator
Dhidden = 256  # hidden units of Discriminator's network
Ghidden = 512  # hidden units of Generator's network
K = 8          # maxout units of Discriminator

mini_batch_size = 50
epoch = 50
samples=(5,6)  # samples drawing size

mnist = input_data.read_data_sets("data/", one_hot=True)
N, num_features = mnist.train.images.shape
period = N // mini_batch_size

X = tf.placeholder(tf.float32, shape=(None, num_features))
Z = tf.placeholder(tf.float32, shape=(None, noise_dim))
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("G"):
    GW1 = tf.Variable(tf.random_normal([noise_dim, Ghidden], stddev=0.1))
    Gb1 = tf.Variable(tf.zeros(Ghidden))
    GW2 = tf.Variable(tf.random_normal([Ghidden, num_features], stddev=0.1))
    Gb2 = tf.Variable(tf.zeros(num_features))

with tf.variable_scope("D"):
    DW1 = tf.Variable(tf.random_normal([num_features, K * Dhidden], stddev=0.01))
    Db1 = tf.Variable(tf.zeros(K * Dhidden))
    DW2 = tf.Variable(tf.random_normal([Dhidden, 1], stddev=0.01))
    Db2 = tf.Variable(tf.zeros(1))

def discriminator(xx):
    u = tf.reshape(tf.matmul(xx, DW1) + Db1, [-1, K, Dhidden])
    Dh1 = tf.nn.dropout(tf.reduce_max(u, reduction_indices=[1]), keep_prob)
    return tf.nn.sigmoid(tf.matmul(Dh1, DW2) + Db2)

Gh1 = tf.nn.relu(tf.matmul(Z, GW1) + Gb1)
G = tf.nn.sigmoid(tf.matmul(Gh1, GW2) + Gb2)
DG = discriminator(G)
Dloss = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - DG))
Gloss = tf.reduce_mean(tf.log(1 - DG) - tf.log(DG + 1e-9)) # the second term for stable learning

vars = tf.trainable_variables()
Dvars = [v for v in vars if v.name.startswith("D")]
Gvars = [v for v in vars if v.name.startswith("G")]

Doptimizer = tf.train.AdamOptimizer().minimize(Dloss, var_list=Dvars)
Goptimizer = tf.train.AdamOptimizer().minimize(Gloss, var_list=Gvars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

dloss = gloss = 0.0
for i in range(epoch * period):
    x, _ = mnist.train.next_batch(mini_batch_size)
    z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
    loss, _ = sess.run([Dloss, Doptimizer], feed_dict={X:x, Z:z, keep_prob:0.5})
    dloss += loss
    z = numpy.random.uniform(-1, 1, size=(mini_batch_size, noise_dim))
    loss, _ = sess.run([Gloss, Goptimizer], feed_dict={Z:z, keep_prob:1.0})
    gloss += loss

    if (i+1) % period == 0:
        print("%d: dloss=%.5f, gloss=%.5f" % ((i+1)//period, dloss / period, gloss / period))
        if math.isnan(dloss) or math.isnan(gloss):
            sess.run(tf.initialize_all_variables()) # initialize & retry if NaN
        dloss = gloss = 0.0

nsamples = samples[0] * samples[1]
def save_figures(path, z):
    Gz = sess.run(G, feed_dict={Z: z})
    fig = plt.figure()
    fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    for i in range(nsamples):
        ax = fig.add_subplot(samples[0], samples[1], i + 1)
        ax.axis("off")
        ax.imshow(Gz[i,:].reshape((28,28)), cmap=plt.get_cmap("gray"))
    fig.savefig(path)
    plt.close(fig)

z = numpy.random.uniform(-1, 1, size=(nsamples, noise_dim))
#z[:,0] = numpy.arange(0, nsamples) / (nsamples - 1) * 2 - 1
save_figures("gan-mnist.png", z)

