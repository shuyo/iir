#!/usr/bin/env python
# encode: utf-8

# Semi-Supervised Naive Bayes Classifier with EM-Algorithm
#    [K. Nigam, A. McCallum, S. Thrun, and T. Mitchell 2000] Text Classifcation from Labeled and Unlabeled Documents using EM. Machine Learning

# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.


import optparse
import numpy, scipy
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer

def performance(i, test, phi, theta):
    z = test.data * numpy.log(phi) + numpy.log(theta) # M * K
    z -= z.max(axis=1)[:, None]
    z = numpy.exp(z)
    z /= z.sum(axis=1)[:, None]
    predict = z.argmax(axis=1)
    correct = (test.target == predict).sum()
    T = test.data.shape[0]
    accuracy = float(correct) / T
    log_likelihood = numpy.log(numpy.choose(test.target, z.T) + 1e-14).sum() / T

    print "%d : %d / %d = %.3f, average of log likelihood = %.3f" % (i, correct, T, accuracy, log_likelihood)
    return accuracy

def estimate(data, test, alpha, beta, n, K=None):
    M, V = data.data.shape
    if not K:
        K = data.target.max() + 1
    #if opt.training:
    #    train = [int(x) for x in opt.training.split(",")]
    #else:
    train = []
    for k in xrange(K):
        train.extend(numpy.random.choice((data.target==k).nonzero()[0], n))

    theta = numpy.ones(K) / K
    phi0 = numpy.zeros((V, K)) + beta
    for n in train:
        phi0[:, data.target[n]] += data.data[n, :].toarray().flatten()
    phi = phi0 / phi0.sum(axis=0)
    accuracy0 = performance(0, test, phi, theta)

    for i in xrange(20):
        # E-step
        z = data.data * numpy.log(phi) + numpy.log(theta) # M * K
        z -= z.max(axis=1)[:, None]
        z = numpy.exp(z)
        z /= z.sum(axis=1)[:, None]

        # M-step
        theta = z.sum(axis=0) + alpha
        theta /= theta.sum()
        phi = phi0 + data.data.T * z
        phi = phi / phi.sum(axis=0)

        accuracy = performance(i+1, test, phi, theta)

    return len(train), accuracy0, accuracy

def main():
    parser = optparse.OptionParser()

    parser.add_option("-K", dest="class_size", type="int", help="number of class")
    parser.add_option("-a", dest="alpha", type="float", help="parameter alpha", default=0.05)
    parser.add_option("-b", dest="beta", type="float", help="parameter beta", default=0.001)
    #parser.add_option("-n", dest="n", type="int", help="training size for each label", default=1)
    #parser.add_option("-t", dest="training", help="specify indexes of training", default=None)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()
    numpy.random.seed(opt.seed)

    data = sklearn.datasets.fetch_20newsgroups()
    test = sklearn.datasets.fetch_20newsgroups(subset='test')

    vec = CountVectorizer()
    data.data = vec.fit_transform(data.data).tocsr()
    test.data = vec.transform(test.data).tocsr() # use the same vocaburary of training data

    print "(data size, voca size) : (%d, %d)" % data.data.shape
    print "(test size, voca size) : (%d, %d)" % test.data.shape

    if opt.class_size:
        """
        index = data.target < opt.class_size
        a = data.data.toarray()[index, :]
        data.data = scipy.sparse.csr_matrix(a)
        data.target = data.target[index]
        print "(shrinked data size, voca size) : (%d, %d)" % data.data.shape
        """

        index = test.target < opt.class_size
        a = test.data.toarray()[index, :]
        test.data = scipy.sparse.csr_matrix(a)
        test.target = test.target[index]
        print "(shrinked test size, voca size) : (%d, %d)" % test.data.shape


    result = []
    for n in xrange(50):
        result.append(estimate(data, test, opt.alpha, opt.beta, n+1, 2))
    for x in result:
        print x

if __name__ == "__main__":
    main()

