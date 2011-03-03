#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + Collapsed Variational Bayes
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy

class LDA_CVB0:
    def __init__(self, K, alpha, beta, docs, V):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V

        self.gamma_jik = []
        self.n_wk = numpy.zeros((self.V, self.K)) + beta
        self.n_jk = numpy.zeros((len(self.docs), self.K)) + alpha
        self.n_k = numpy.zeros(self.K) + V * beta
        self.N = 0
        for j, doc in enumerate(docs):
            N = len(doc)
            self.N += N

            # sample from Dirichlet(alpha) for every words
            gamma_ik = numpy.random.gamma(beta, 1, N * K).reshape(K, N)
            gamma_ik = (gamma_ik / gamma_ik.sum(axis=0)).transpose()
            self.gamma_jik.append(gamma_ik)

            for w, gamma_k in zip(doc, gamma_ik):
                self.n_wk[w] += gamma_k
                self.n_jk[j] += gamma_k
                self.n_k += gamma_k

    def inference(self):
        """learning once iteration"""
        for j, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                gamma_k = self.gamma_jik[j][i]
                self.n_wk[w] -= gamma_k
                self.n_jk[j] -= gamma_k
                self.n_k -= gamma_k
                new_gamma_k = self.n_wk[w] * self.n_jk[j] / self.n_k
                new_gamma_k /= new_gamma_k.sum()
                self.gamma_jik[j][i] = new_gamma_k
                self.n_wk[w] += new_gamma_k
                self.n_jk[j] += new_gamma_k
                self.n_k += new_gamma_k

    def worddist(self):
        """get topic-word distribution"""
        return self.n_wk / self.n_k

    def perplexity(self):
        phi = self.worddist()
        log_per = 0
        for j, doc in enumerate(self.docs):
            theta = self.n_jk[j].copy()
            theta /= theta.sum()
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[w], theta))
        return numpy.exp(log_per / self.N)

def lda_learning(lda, iteration):
    for i in range(iteration):
        print "-%d p=%f" % (i + 1, lda.perplexity())
        lda.inference()
    print "perplexity=%f" % lda.perplexity()

def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="stopwords", type="int", help="0=exclude stop words, 1=include stop words, 2=stop words into one topic", default=1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        numpy.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords==0)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA_CVB0(options.K, options.alpha, options.beta, docs, voca.size())
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration)

    phi = lda.worddist()
    for k in range(options.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[:,k])[:20]:
            print "%s: %f" % (voca[w], phi[w,k])

if __name__ == "__main__":
    main()
