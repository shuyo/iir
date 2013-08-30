#!/usr/bin/env python
# encode: utf-8

# Active Learning (Uncertainly Sampling and Information Density) for 20 newsgroups
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import optparse
import numpy
import scipy.sparse
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def activelearn(results, data, test, strategy, train, pool, classifier_factory, max_train, densities):
    # copy initial indexes of training and pool
    train = list(train)
    pool = list(pool)

    accuracies = []
    while len(train) < max_train:
        if len(accuracies) > 0:
            if strategy == "random":
                x = numpy.random.randint(len(pool))
            else:
                predict = cl.predict_proba(data.data[pool,:])
                if strategy == "least confident":
                    x = predict.max(axis=1)-1
                elif strategy == "margin sampling":
                    predict.sort(axis=1)
                    x = (predict[:,-1] - predict[:,-2])
                elif strategy == "entropy-based":
                    x = numpy.nan_to_num(predict * numpy.log(predict)).sum(axis=1)
                if densities != None: x *= densities[pool]
                x = x.argmin()
            train.append(pool[x])
            del pool[x]

        cl = classifier_factory()
        cl.fit(data.data[train,:], data.target[train])
        accuracy = cl.score(test.data, test.target)
        print "%d : %f" % (len(train), accuracy)
        accuracies.append(accuracy)

    results.append((strategy, accuracies))


def main():
    parser = optparse.OptionParser()
    parser.add_option("--nb", dest="naive_bayes", type="float", help="use naive bayes classifier", default=None)
    parser.add_option("--lr1", dest="logistic_l1", type="float", help="use logistic regression with l1-regularity", default=None)
    parser.add_option("--lr2", dest="logistic_l2", type="float", help="use logistic regression with l2-regularity", default=None)

    parser.add_option("-K", dest="class_size", type="int", help="number of class", default=None)
    parser.add_option("-n", dest="max_train", type="int", help="max size of training", default=100)
    parser.add_option("-N", dest="trying", type="int", help="number of trying", default=100)

    parser.add_option("-b", dest="beta", type="float", help="density importance", default=0)
    (opt, args) = parser.parse_args()

    data = sklearn.datasets.fetch_20newsgroups_vectorized()
    print "(train size, voca size) : (%d, %d)" % data.data.shape

    if opt.class_size:
        index = data.target < opt.class_size
        a = data.data.toarray()[index, :]
        data.data = scipy.sparse.csr_matrix(a)
        data.target = data.target[index]
        print "(shrinked train size, voca size) : (%d, %d)" % data.data.shape

    classifier_factory = clz = None
    if opt.logistic_l1:
        print "Logistic Regression with L1-regularity : C = %f" % opt.logistic_l1
        classifier_factory = lambda: LogisticRegression(penalty='l1', C=opt.logistic_l1)
        clz = "lrl1"
    elif opt.logistic_l2:
        print "Logistic Regression with L2-regularity : C = %f" % opt.logistic_l2
        classifier_factory = lambda: LogisticRegression(C=opt.logistic_l2)
        clz = "lrl2"
    elif opt.naive_bayes:
        print "Naive Bayes Classifier : alpha = %f" % opt.naive_bayes
        classifier_factory = lambda: MultinomialNB(alpha=opt.naive_bayes)
        clz = "nb"

    if classifier_factory:
        test = sklearn.datasets.fetch_20newsgroups_vectorized(subset='test')
        print "(test size, voca size) : (%d, %d)" % test.data.shape
        if opt.class_size:
            index = test.target < opt.class_size
            a = test.data.toarray()[index, :]
            test.data = scipy.sparse.csr_matrix(a)
            test.target = test.target[index]
            print "(shrinked test size, voca size) : (%d, %d)" % test.data.shape

        densities = None
        if opt.beta > 0:
            densities = (data.data * data.data.T).mean(axis=0).A[0] ** opt.beta

        N_CLASS = data.target.max() + 1
        for method in ["random", "least confident", "margin sampling", "entropy-based"]:
            results = []
            for n in xrange(opt.trying):
                print "%s : %d" % (method, n)
                train = [numpy.random.choice((data.target==k).nonzero()[0]) for k in xrange(N_CLASS)]
                pool = range(data.data.shape[0])
                for x in train: pool.remove(x)

                activelearn(results, data, test, method, train, pool, classifier_factory, opt.max_train, densities)

            d = len(train)
            with open("output_%s_%s.txt" % (method, clz), "wb") as f:
                f.write(method)
                f.write("\n")
                for i in xrange(len(results[0][1])):
                    f.write("%d\t%s\n" % (i+d, "\t".join("%f" % x[1][i] for x in results)))


if __name__ == "__main__":
    main()
