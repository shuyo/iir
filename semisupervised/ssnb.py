#!/usr/bin/env python
# encode: utf-8

# Semi-Supervised Naive Bayes Classifier with EM-Algorithm
#    [K. Nigam, A. McCallum, S. Thrun, and T. Mitchell 2000] Text Classifcation from Labeled and Unlabeled Documents using EM. Machine Learning

# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.


import optparse
import numpy
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer

def main():
    parser = optparse.OptionParser()
    """
    parser.add_option("-r", dest="method_random", action="store_true", help="use random sampling", default=False)
    parser.add_option("-n", dest="max_train", type="int", help="max size of training", default=300)
    """

    parser.add_option("-a", dest="alpha", type="float", help="parameter alpha", default=0.05)
    parser.add_option("-b", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-t", dest="training", help="specify indexes of training", default=None)
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

    M, V = data.data.shape
    T = test.data.shape[0]
    K = data.target.max() + 1
    if opt.training:
        train = [int(x) for x in opt.training.split(",")]
    else:
        train = [numpy.random.choice((data.target==k).nonzero()[0]) for k in xrange(K)]

    theta = numpy.ones(K) / K
    phi0 = numpy.zeros((V, K)) + opt.beta
    for n in train:
        phi0[:, data.target[n]] += data.data[n, :].toarray().flatten()
    phi = phi0 / phi0.sum(axis=1)[:, None]

    for i in xrange(20):
        # E-step
        z = data.data * numpy.log(phi) + numpy.log(theta) # M * K
        z -= z.max(axis=1)[:, None]
        z = numpy.exp(z)
        z /= z.sum(axis=1)[:, None]

        # M-step
        theta = z.sum(axis=0) + opt.alpha
        theta /= theta.sum()
        phi = phi0 + data.data.T * z
        phi = phi / phi.sum(axis=1)[:, None]

        # predict
        z = test.data * numpy.log(phi) + numpy.log(theta) # M * K
        z -= z.max(axis=1)[:, None]
        z = numpy.exp(z)
        z /= z.sum(axis=1)[:, None]
        predict = z.argmax(axis=1)
        correct = (test.target == predict).sum()
        log_likelihood = -numpy.log(numpy.choose(test.target, z.T)).sum()

        print "%d : %d / %d = %.3f, log likelihood = %.3f" % (i, correct, T, float(correct) / T, log_likelihood)

if __name__ == "__main__":
    main()

