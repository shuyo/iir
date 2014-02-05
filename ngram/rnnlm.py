#!/usr/bin/env python
# encode: utf-8

# Recurrent Neural Network Langage Model
# This code is available under the MIT License.
# (c)2014 Nakatani Shuyo / Cybozu Labs Inc.

import sys, codecs, re
import numpy, nltk
import optparse

class RNNLM:
    def __init__(self, K, V):
        self.K = K
        self.v = V
        self.U = numpy.random.randn(K, V)
        self.W = numpy.random.randn(K, K)
        self.V = numpy.random.randn(V, K)
    def learn(self, docs, alpha=0.1):
        for doc in docs:
            pre_s = numpy.zeros(self.K)
            pre_w = 0
            for w in doc:
                s = 1 / (numpy.exp(- numpy.dot(self.W, pre_s) - self.U[:, pre_w]) + 1)
                pre_w = w
                z = numpy.dot(self.V, s)
                y = numpy.exp(z - z.max())
                y = y / y.sum()
                y[w] -= 1  # -e0
                self.V -= numpy.outer(y, s * alpha)
                meh = numpy.dot(y, self.V) * alpha # -eh * alpha
                self.U[:, w] += meh * s * (s - 1)
                self.W -= numpy.outer(pre_s, meh)

    def perplexity(self, docs):
        log_like = 0
        N = 0
        for doc in docs:
            s = numpy.zeros(self.K)
            pre_w = 0
            for w in doc:
                s = 1 / (numpy.exp(- numpy.dot(self.W, s) - self.U[:, pre_w]) + 1)
                pre_w = w
                z = numpy.dot(self.V, s)
                y = numpy.exp(z - z.max())
                y = y / y.sum()
                log_like -= numpy.log(y[w])
            N += len(doc)
        return log_like / N

def main():
    parser = optparse.OptionParser()
    parser.add_option("-c", dest="corpus", help="corpus module name under nltk.corpus (e.g. brown, reuters)", default='nps_chat')
    parser.add_option("-r", dest="testrate", type="float", help="rate of test dataset in corpus", default=0.1)
    parser.add_option("-k", dest="K", type="int", help="size of hidden layer", default=10)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()

    numpy.random.seed(opt.seed)

    m = __import__('nltk.corpus', globals(), locals(), [opt.corpus], -1)
    corpus = getattr(m, opt.corpus)
    ids = corpus.fileids()
    D = len(ids)
    print "found corpus : %s (D=%d)" % (opt.corpus, D)

    voca = {"<s>":0, "</s>":1}
    vocalist = ["<s>", "</s>"]
    docs = []
    for id in corpus.fileids():
        doc = [0]
        for w in corpus.words(id):
            w = w.lower()
            if w not in voca:
                voca[w] = len(vocalist)
                vocalist.append(w)
            doc.append(voca[w])
        if len(doc) > 1:
            doc.append(1)
            docs.append(doc)
    V = len(vocalist)
    print "vocaburary : %d / %d" % (V, len(corpus.words()))

    D = len(docs)

    model = RNNLM(opt.K, V)
    print model.perplexity(docs)
    for i in xrange(10):
        model.learn(docs)
        print model.perplexity(docs)

"""
    testids = set(random.sample(ids, int(D * opt.testrate)))
    trainids = [id for id in ids if id not in testids]
    trainwords = [w.lower() for w in corpus.words(trainids)]

    testset = []
    voca = set(freq1.iterkeys())
    for id in testids:
        f = corpus.words(id)
        doc = [w.lower() for w in f]
        f.close()

        testset.append(doc)
        for w in doc:
            voca.add(w)
"""
if __name__ == "__main__":
    main()
