#!/usr/bin/env python
# encode: utf-8

# Active Learning (Uncertainly Sampling)
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import dataset
from sklearn.linear_model import LogisticRegression

categories = ['crude', 'money-fx', 'trade', 'interest', 'ship', 'wheat', 'corn']
doclist, labels, voca, vocalist = dataset.load(categories)
print "document size : %d" % len(doclist)
print "vocaburary size : %d" % len(voca)

data = numpy.zeros((len(doclist), len(voca)))
for j, doc in enumerate(doclist):
    for i, c in doc.iteritems():
        data[j, i] = c

def activelearn(data, label, strategy, train):
    print strategy

    N, D = data.shape
    train = list(train) # copy initial indexes of training
    pool = range(N)
    for x in train: pool.remove(x)

    predict = None
    precisions = []
    while len(train) < 300:
        if predict != None:
            if strategy == "random":
                x = numpy.random.randint(len(pool))
            elif strategy == "least confident":
                x = predict.max(axis=1).argmin()
            elif strategy == "margin sampling":
                predict.sort(axis=1)
                x = (numpy.exp(predict[:,-1])-numpy.exp(predict[:,-2])).argmin()
            elif strategy == "entropy-based":
                x = numpy.nan_to_num(numpy.exp(predict)*predict).sum(axis=1).argmin()
            train.append(pool[x])
            del pool[x]

        cl = LogisticRegression()
        cl.fit(data[train,:], label[train])
        predict = cl.predict_log_proba(data[pool,:])
        log_likelihood = 0
        correct = 0
        for n, logprob in zip(pool,predict):
            c = label[n]
            log_likelihood += logprob[c]
            if c == logprob.argmax(): correct += 1

        Z = len(pool)
        precision = float(correct) / Z
        perplexity = numpy.exp(-log_likelihood / Z)
        print "%d : %d / %d = %f, %f" % (len(train), correct, Z, precision, perplexity)

        precisions.append(precision)

    return precisions

N_CLASS = labels.max() + 1
train = [numpy.random.choice((labels==k).nonzero()[0]) for k in xrange(N_CLASS)]

methods = ["random", "least confident", "margin sampling", "entropy-based"]
results = []
for x in methods:
    results.append(activelearn(data, labels, x, train))
print "\t%s" % "\t".join(methods)
d = len(categories)
for i in xrange(len(results[0])):
    print "%d\t%s" % (i+d, "\t".join("%f" % x[i] for x in results))
