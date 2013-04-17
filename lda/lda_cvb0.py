#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + Collapsed Variational Bayesian
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy

class LDA_CVB0:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.V = V

        self.docs = []
        self.gamma_jik = []
        self.n_wk = numpy.zeros((V, K)) + beta
        self.n_jk = numpy.zeros((len(docs), K)) + alpha
        self.n_k = numpy.zeros(K) + V * beta
        self.N = 0
        for j, doc in enumerate(docs):
            self.N += len(doc)
            term_freq = dict()
            term_gamma = dict()
            for i, w in enumerate(doc):
                if smartinit:
                    p_k = self.n_wk[w] * self.n_jk[j] / self.n_k
                    gamma_k = numpy.random.mtrand.dirichlet(p_k / p_k.sum() * alpha)
                else:
                    gamma_k = [float("nan")]
                if not numpy.isfinite(gamma_k[0]): # maybe NaN or Inf
                    gamma_k = numpy.random.mtrand.dirichlet([alpha] * K)
                if w in term_freq:
                    term_freq[w] += 1
                    term_gamma[w] += gamma_k
                else:
                    term_freq[w] = 1
                    term_gamma[w] = gamma_k
                self.n_wk[w] += gamma_k
                self.n_jk[j] += gamma_k
                self.n_k += gamma_k
            term_freq = term_freq.items()
            self.docs.append(term_freq)
            self.gamma_jik.append([term_gamma[w] / freq for w, freq in term_freq])

    def inference(self):
        """learning once iteration"""
        new_n_wk = numpy.zeros((self.V, self.K)) + self.beta
        new_n_jk = numpy.zeros((len(self.docs), self.K)) + self.alpha
        n_k = self.n_k
        for j, doc in enumerate(self.docs):
            gamma_ik = self.gamma_jik[j]
            n_jk = self.n_jk[j]
            new_n_jk_j = new_n_jk[j]
            for i, gamma_k in enumerate(gamma_ik):
                w, freq = doc[i]
                new_gamma_k = (self.n_wk[w] - gamma_k) * (n_jk - gamma_k) / (n_k - gamma_k)
                new_gamma_k /= new_gamma_k.sum()

                gamma_ik[i] = new_gamma_k
                gamma_freq = new_gamma_k * freq
                new_n_wk[w] += gamma_freq
                new_n_jk_j += gamma_freq

        self.n_wk = new_n_wk
        self.n_jk = new_n_jk
        self.n_k  = new_n_wk.sum(axis=0)

    def worddist(self):
        """get topic-word distribution"""
        return numpy.transpose(self.n_wk / self.n_k)

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        for j, doc in enumerate(docs):
            theta = self.n_jk[j]
            theta = theta / theta.sum()
            for w, freq in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta)) * freq
                N += freq
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    phi = lda.worddist()
    for k in range(lda.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %f" % (voca[w], phi[k,w])

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
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
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

    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA_CVB0(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print "corpus=%d, words=%d, voca=%d, K=%d, a=%f, b=%f" % (len(corpus), lda.N, len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

if __name__ == "__main__":
    main()
