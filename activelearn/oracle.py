#!/usr/bin/env python
# encode: utf-8

# Active Learning for 20 newsgroups with Oracle and testset

# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import optparse
import numpy
import scipy.sparse
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def activelearn(data, test, train, pool, classifier_factory, max_train, n_candidate, seed):
    numpy.random.seed(seed)

    # copy initial indexes of training and pool
    train = list(train)
    pool = list(pool)

    accuracies = []
    Z = len(test.target)
    K = data.target.max() + 1
    while len(train) < max_train:
        if len(accuracies) > 0:
            i_star = None
            max_score = 0.0
            candidate = pool
            if 0 < n_candidate < len(pool):
                numpy.random.shuffle(pool)
                candidate = pool[:n_candidate]
            for i, x in enumerate(candidate):
                t = train + [x]
                s = classifier_factory().fit(data.data[t, :], data.target[t]).score(test.data, test.target)
                if max_score < s:
                    print "%d\t%f" % (x, s)
                    max_score = s
                    i_star = i
            train.append(pool[i_star])
            del pool[i_star]

        classifier = classifier_factory().fit(data.data[train,:], data.target[train])
        accuracy = classifier.score(test.data, test.target)
        print "%d : %f" % (len(train), accuracy)
        accuracies.append((len(train), accuracy))

    return accuracies

def main():
    parser = optparse.OptionParser()
    parser.add_option("--nb", dest="naive_bayes", type="float", help="use naive bayes classifier", default=None)
    parser.add_option("--lr1", dest="logistic_l1", type="float", help="use logistic regression with l1-regularity", default=None)
    parser.add_option("--lr2", dest="logistic_l2", type="float", help="use logistic regression with l2-regularity", default=None)

    parser.add_option("-K", dest="class_size", type="int", help="number of class", default=None)
    parser.add_option("-n", dest="max_train", type="int", help="max size of training", default=30)
    parser.add_option("-t", dest="training", help="specify indexes of training", default=None)
    parser.add_option("-T", dest="candidate", type="int", help="candidate size", default=-1)

    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()

    data = sklearn.datasets.fetch_20newsgroups_vectorized()
    print "(train size, voca size) : (%d, %d)" % data.data.shape

    if opt.class_size:
        index = data.target < opt.class_size
        a = data.data.toarray()[index, :]
        data.data = scipy.sparse.csr_matrix(a)
        data.target = data.target[index]
        print "(shrinked train size, voca size) : (%d, %d)" % data.data.shape


    N_CLASS = data.target.max() + 1
    if opt.training:
        train = [int(x) for x in opt.training.split(",")]
    else:
        train = [numpy.random.choice((data.target==k).nonzero()[0]) for k in xrange(N_CLASS)]
    print "indexes of training set : ", ",".join("%d" % x for x in train)

    pool = range(data.data.shape[0])
    for x in train: pool.remove(x)

    classifier_factory = None
    if opt.logistic_l1:
        print "Logistic Regression with L1-regularity : C = %f" % opt.logistic_l1
        classifier_factory = lambda: LogisticRegression(penalty='l1', C=opt.logistic_l1)
    elif opt.logistic_l2:
        print "Logistic Regression with L2-regularity : C = %f" % opt.logistic_l2
        classifier_factory = lambda: LogisticRegression(C=opt.logistic_l2)
    elif opt.naive_bayes:
        print "Naive Bayes Classifier : alpha = %f" % opt.naive_bayes
        classifier_factory = lambda: MultinomialNB(alpha=opt.naive_bayes)

    if classifier_factory:
        test = sklearn.datasets.fetch_20newsgroups_vectorized(subset='test')
        print "(test size, voca size) : (%d, %d)" % test.data.shape
        if opt.class_size:
            index = test.target < opt.class_size
            a = test.data.toarray()[index, :]
            test.data = scipy.sparse.csr_matrix(a)
            test.target = test.target[index]
            print "(shrinked test size, voca size) : (%d, %d)" % test.data.shape

        print "score for all data: %f" % classifier_factory().fit(data.data, data.target).score(test.data, test.target)

        results = activelearn(data, test, train, pool, classifier_factory, opt.max_train, opt.candidate, opt.seed)

        for x in results:
            print "%d\t%f" % x

if __name__ == "__main__":
    main()
