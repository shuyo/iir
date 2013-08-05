#!/usr/bin/env python
# encode: utf-8

# Active Learning (Uncertainly Sampling)
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import re, collections, numpy
from nltk.corpus import reuters
from nltk.stem import WordNetLemmatizer

categories = ['crude', 'money-fx', 'trade', 'interest', 'ship', 'wheat', 'corn']

map = dict()
intersection = set()
for x, catname in enumerate(categories):
    for id in reuters.fileids(catname):
        if id in map:
            intersection.add(id)
        else:
            map[id] = x
for id in intersection:
    del map[id]
fileids = map.keys()
labels = numpy.array(map.values())

voca = dict()
vocalist = []
doclist = []
realphabet = re.compile('^[a-z]+$')
wnl = WordNetLemmatizer()
for id in fileids:
    doc = collections.defaultdict(int)
    for w in reuters.words(id):
        w = wnl.lemmatize(w.lower())
        if realphabet.match(w):
            if w not in voca:
                voca[w] = len(vocalist)
                vocalist.append(w)
            doc[voca[w]] += 1
    if len(doc) == 0: print id
    doclist.append(doc)
print "document size : %d" % len(doclist)
print "vocaburary size : %d" % len(voca)


data = numpy.zeros((len(doclist), len(voca)))
for j, doc in enumerate(doclist):
    for i, c in doc.iteritems():
        data[j, i] = c

def activelearn(data, label, strategy, train):
    print strategy

    N, D = data.shape
    train = list(train)
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
        #cl = LogisticRegression(C=0.1, penalty="l1")
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

from sklearn.linear_model import LogisticRegression
cl = LogisticRegression()

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
