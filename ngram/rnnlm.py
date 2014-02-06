#!/usr/bin/env python
# encode: utf-8

# Recurrent Neural Network Language Model
# This code is available under the MIT License.
# (c)2014 Nakatani Shuyo / Cybozu Labs Inc.

import numpy, nltk, codecs, re
import optparse

class RNNLM:
    def __init__(self, V, K=10):
        self.K = K
        self.v = V
        self.U = numpy.random.randn(K, V) / 3
        self.W = numpy.random.randn(K, K) / 3
        self.V = numpy.random.randn(V, K) / 3

    def learn(self, docs, alpha=0.1):
        index = numpy.arange(len(docs))
        numpy.random.shuffle(index)
        for i in index:
            doc = docs[i]
            pre_s = numpy.zeros(self.K)
            pre_w = 0 # <s>
            for w in doc:
                s = 1 / (numpy.exp(- numpy.dot(self.W, pre_s) - self.U[:, pre_w]) + 1)
                z = numpy.dot(self.V, s)
                y = numpy.exp(z - z.max())
                y = y / y.sum()
                y[w] -= 1  # -e0
                eha = numpy.dot(y, self.V) * s * (s - 1) * alpha # eh * alpha
                self.V -= numpy.outer(y, s * alpha)
                self.U[:, pre_w] += eha
                self.W += numpy.outer(pre_s, eha)
                pre_w = w
                pre_s = s

    def perplexity(self, docs):
        log_like = 0
        N = 0
        for doc in docs:
            s = numpy.zeros(self.K)
            pre_w = 0 # <s>
            for w in doc:
                s = 1 / (numpy.exp(- numpy.dot(self.W, s) - self.U[:, pre_w]) + 1)
                z = numpy.dot(self.V, s)
                y = numpy.exp(z - z.max())
                y = y / y.sum()
                log_like -= numpy.log(y[w])
                pre_w = w
            N += len(doc)
        return log_like / N

    def dist(self, w):
        if w==0:
            self.s = numpy.zeros(self.K)
        else:
            self.s = 1 / (numpy.exp(- numpy.dot(self.W, self.s) - self.U[:, w]) + 1)
            z = numpy.dot(self.V, self.s)
            y = numpy.exp(z - z.max())
            return y / y.sum()

class RNNLM_BPTT(RNNLM):
    """RNNLM with BackPropagation Through Time"""
    def learn(self, docs, alpha=0.1, tau=3):
        index = numpy.arange(len(docs))
        numpy.random.shuffle(index)
        for i in index:
            doc = docs[i]
            pre_s = [numpy.zeros(self.K)]
            pre_w = [0] # <s>
            for w in doc:
                s = 1 / (numpy.exp(- numpy.dot(self.W, pre_s[-1]) - self.U[:, pre_w[-1]]) + 1)
                z = numpy.dot(self.V, s)
                y = numpy.exp(z - z.max())
                y = y / y.sum()

                # calculate errors
                y[w] -= 1  # -e0
                eh = [numpy.dot(y, self.V) * s * (s - 1)] # eh[t]
                for t in xrange(min(tau, len(pre_s)-1)):
                    st = pre_s[-1-t]
                    eh.append(numpy.dot(eh[-1], self.W) * st * (1 - st))

                # update parameters
                pre_w.append(w)
                pre_s.append(s)
                self.V -= numpy.outer(y, s * alpha)
                for t in xrange(len(eh)):
                    self.U[:, pre_w[-1-t]] += eh[t] * alpha
                    self.W += numpy.outer(pre_s[-2-t], eh[t]) * alpha

class BIGRAM:
    def __init__(self, V, alpha=0.01):
        self.V = V
        self.alpha = alpha
        self.count = dict()
        self.amount = numpy.zeros(V, dtype=int)
    def learn(self, docs):
        for doc in docs:
            pre_w = 0 # <s>
            for w in doc:
                if pre_w not in self.count:
                    self.count[pre_w] = {w:1}
                elif w not in self.count[pre_w]:
                    self.count[pre_w][w] = 1
                else:
                    self.count[pre_w][w] += 1
                self.amount[pre_w] += 1
                pre_w = w

    def perplexity(self, docs):
        log_like = 0
        N = 0
        va = self.V * self.alpha
        for doc in docs:
            pre_w = 0 # <s>
            for w in doc:
                c = 0
                if pre_w in self.count and w in self.count[pre_w]:
                    c = self.count[pre_w][w]
                log_like -= numpy.log((c + self.alpha) / (self.amount[pre_w] + va))
                pre_w = w
            N += len(doc)
        return log_like / N

def CorpusWrapper(corpus):
    for id in corpus.fileids():
        yield corpus.words(id)

def main():
    parser = optparse.OptionParser()
    parser.add_option("-c", dest="corpus", help="corpus module name under nltk.corpus (e.g. brown, reuters)")
    parser.add_option("-f", dest="filename", help="corpus filename name (each line is regarded as a document)")
    parser.add_option("-a", dest="alpha", type="float", help="additive smoothing parameter of bigram", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="size of hidden layer", default=10)
    parser.add_option("-i", dest="I", type="int", help="learning interval", default=10)
    parser.add_option("-o", dest="output", help="output filename of rnnlm model")
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()

    numpy.random.seed(opt.seed)

    if opt.corpus:
        m = __import__('nltk.corpus', globals(), locals(), [opt.corpus], -1)
        corpus = CorpusWrapper(getattr(m, opt.corpus))
    elif opt.filename:
        corpus = []
        with codecs.open(opt.filename, "rb", "utf-8") as f:
            for s in f:
                s = re.sub(r'(["\.,!\?:;])', r' \1 ', s).strip()
                d = re.split(r'\s+', s)
                if len(d) > 0: corpus.append(d)
    else:
        raise "need -f or -c"

    voca = {"<s>":0, "</s>":1}
    vocalist = ["<s>", "</s>"]
    docs = []
    N = 0
    for words in corpus:
        doc = []
        for w in words:
            w = w.lower()
            if w not in voca:
                voca[w] = len(vocalist)
                vocalist.append(w)
            doc.append(voca[w])
        if len(doc) > 0:
            N += len(doc)
            doc.append(1) # </s>
            docs.append(doc)

    D = len(docs)
    V = len(vocalist)
    print "corpus : %s (D=%d)" % (opt.corpus or opt.filename, D)
    print "vocabulary : %d / %d" % (V, N)

    print ">> RNNLM(K=%d)" % opt.K
    model = RNNLM_BPTT(V, opt.K)
    a = 1.0
    for i in xrange(opt.I):
        print i, model.perplexity(docs)
        model.learn(docs, a)
        a = a * 0.95 + 0.01
    print opt.I, model.perplexity(docs)

    if opt.output:
        import cPickle
        with open(opt.output, 'wb') as f:
            cPickle.dump([model, voca, vocalist], f)

    print ">> BIGRAM(alpha=%f)" % opt.alpha
    model = BIGRAM(V, opt.alpha)
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
