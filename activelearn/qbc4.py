#!/usr/bin/env python
# encode: utf-8

# Active Learning (Query-By-Committee) for 20 newsgroups
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import optparse
import numpy
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def activelearn(results, data, test, strategy, train, pool, classifier_factories, max_train, densities):
    print strategy

    # copy initial indexes of training and pool
    train = list(train)
    pool = list(pool)

    accuracies = []
    Z = len(test.target)
    while len(train) < max_train:
        if len(accuracies) > 0:
            if strategy == "random":
                x = numpy.random.randint(len(pool))
            else:
                if strategy == "vote entropy":
                    p = numpy.array([c.predict(data.data[pool,:]) for c in classifiers])
                    # This is equivalent to Vote Entropy when # of classifiers = 3
                    x = ((p[:,0:2]==p[:,1:3]).sum(axis=1) + (p[:,0]==p[:,2]))
                elif strategy == "average KL":
                    p = numpy.array([c.predict_proba(data.data[pool,:]) for c in classifiers]) # 3 * N * K
                    pc = p.mean(axis=0) # N * K
                    x = numpy.nan_to_num(p * numpy.log(pc / p)).sum(axis=2).sum(axis=0)
                elif strategy == "qbc+margin sampling":
                    p = numpy.array([c.predict_proba(data.data[pool,:]) for c in classifiers]) # 3 * N * K
                    pc = p.mean(axis=0) # N * K
                    pc.sort(axis=1)
                    x = pc[:,-1] - pc[:,-2]
                if densities != None: x *= densities[pool]
                x = x.argmin()
            train.append(pool[x])
            del pool[x]

        classifiers = [f().fit(data.data[train,:], data.target[train]) for f in classifier_factories]

        predict = sum(c.predict_proba(test.data) for c in classifiers)
        correct = (predict.argmax(axis=1) == test.target).sum()
        accuracy = float(correct) / Z
        print "%s %d : %d / %d = %f" % (strategy, len(train), correct, Z, accuracy)
        accuracies.append(accuracy)

    results.append((strategy, accuracies))

def main():
    parser = optparse.OptionParser()
    parser.add_option("--nb", dest="naive_bayes", type="float", help="use naive bayes classifier", default=None)
    parser.add_option("--lr1", dest="logistic_l1", type="float", help="use logistic regression with l1-regularity", default=None)
    parser.add_option("--lr2", dest="logistic_l2", type="float", help="use logistic regression with l2-regularity", default=None)

    parser.add_option("-n", dest="max_train", type="int", help="max size of training", default=300)
    parser.add_option("-t", dest="training", help="specify indexes of training", default=None)

    parser.add_option("-b", dest="beta", type="float", help="density importance", default=0)

    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()
    numpy.random.seed(opt.seed)

    data = sklearn.datasets.fetch_20newsgroups_vectorized()
    print "(train size, voca size) : (%d, %d)" % data.data.shape

    N_CLASS = data.target.max() + 1
    if opt.training:
        train = [int(x) for x in opt.training.split(",")]
    else:
        train = [numpy.random.choice((data.target==k).nonzero()[0]) for k in xrange(N_CLASS)]
    print "indexes of training set : ", ",".join("%d" % x for x in train)

    pool = range(data.data.shape[0])
    for x in train: pool.remove(x)

    classifier_factories = []
    if opt.logistic_l1:
        print "Logistic Regression with L1-regularity : C = %f" % opt.logistic_l1
        classifier_factories.append(lambda: LogisticRegression(penalty='l1', C=opt.logistic_l1))
    if opt.logistic_l2:
        print "Logistic Regression with L2-regularity : C = %f" % opt.logistic_l2
        classifier_factories.append(lambda: LogisticRegression(C=opt.logistic_l2))
    if opt.naive_bayes:
        print "Naive Bayes Classifier : alpha = %f" % opt.naive_bayes
        classifier_factories.append(lambda: MultinomialNB(alpha=opt.naive_bayes))

    if len(classifier_factories) >= 2:
        test = sklearn.datasets.fetch_20newsgroups_vectorized(subset='test')
        print "(test size, voca size) : (%d, %d)" % test.data.shape

        densities = None
        if opt.beta > 0:
            densities = (data.data * data.data.T).mean(axis=0).A[0] ** opt.beta

        methods = ["random", "vote entropy", "average KL", "qbc+margin sampling", ]
        results = []
        for x in methods:
            activelearn(results, data, test, x, train, pool, classifier_factories, opt.max_train, densities)

        print "\t%s" % "\t".join(x[0] for x in results)
        d = len(train)
        for i in xrange(len(results[0][1])):
            print "%d\t%s" % (i+d, "\t".join("%f" % x[1][i] for x in results))

if __name__ == "__main__":
    main()
