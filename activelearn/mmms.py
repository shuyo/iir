#!/usr/bin/env python
# encode: utf-8

# Active Learning for 20 newsgroups : MCMI[min] with margin sampling
#    MCMI[min] refers to (Guo+ IJCAI-07)
#    Yuhong Guo and Russ Greiner, Optimistic Active Learning using Mutual Information, IJCAI-07

# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import optparse
import numpy
import scipy.sparse
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def activelearn(data, test, train, pool, classifier_factory, max_train, seed):
    numpy.random.seed(seed)

    # copy initial indexes of training and pool
    train = list(train)
    pool = list(pool)

    accuracies = []
    Z = len(test.target)
    K = data.target.max() + 1
    while len(train) < max_train:
        if len(accuracies) > 0:
            predict = classifier.predict_proba(data.data[pool,:])
            predict.sort(axis=1)
            margin = predict[:,-1] - predict[:,-2]
            candidate = margin.argsort()[:30]

            i_star = y_i_star = None
            f_i_star = 1e300
            print "i\ty_i\t(actual)\tf_i\tmargin"
            for i in candidate:
                x = pool[i]
                L_x_i = data.data[train + [x], :]
                L_y = data.target[train]
                entropies = numpy.zeros(K)
                for y in xrange(K):
                    l = list(L_y)
                    l.append(y)
                    phi_i = classifier_factory().fit(L_x_i, l)

                    p = phi_i.predict_proba(data.data[pool])
                    entropies[y] = -(numpy.nan_to_num(numpy.log(p)) * p).sum()
                y_i = entropies.argmin()
                f_i = entropies[y_i]
                print "%d\t%d\t%d\t%f\t%f" % (x, y_i, data.target[x], f_i, margin[i])
                if f_i < f_i_star:
                    i_star = i
                    y_i_star = y_i
                    f_i_star = f_i

            x = pool[i_star]
            print "select : %d (MM=%f, predict=%d, actual=%d)" % (x, f_i_star, y_i_star, data.target[x])
            train.append(x)
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

    parser.add_option("-K", dest="class_size", type="int", help="number of class", default=4)
    parser.add_option("-n", dest="max_train", type="int", help="max size of training", default=100)
    parser.add_option("-t", dest="training", help="specify indexes of training", default=None)
    parser.add_option("-N", dest="trying", type="int", help="number of trying", default=100)

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

        for n in xrange(opt.trying):
            print "trying.. %d" % n
            train = [numpy.random.choice((data.target==k).nonzero()[0]) for k in xrange(N_CLASS)]
            pool = range(data.data.shape[0])
            for x in train: pool.remove(x)
            results = activelearn(data, test, train, pool, classifier_factory, opt.max_train, opt.seed)

            with open("output_mmms_%d_%d.txt" % (opt.class_size, opt.max_train), "ab") as f:
                f.write("\t".join("%f" % x[1] for x in results))
                f.write("\n")

if __name__ == "__main__":
    main()
