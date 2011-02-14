#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import optparse
import numpy
import vocabulary

class LDA:
    def __init__(self, K, alpha, beta):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior

    def set_corpus(self, corpus, stopwords):
        """set courpus and initialize"""
        voca = vocabulary.Vocabulary(stopwords==0)
        self.docs = [voca.doc_to_ids(doc) for doc in corpus]
        self.n_m = numpy.array([len(doc) for doc in self.docs])

        M = len(self.docs)
        self.V = voca.size()

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((M, self.K)) + self.alpha     # word count of each document and topic
        self.n_z_t = numpy.zeros((self.K, self.V)) + self.beta # word count of each topic and vocabulary
        self.n_z = numpy.zeros(self.K) + self.V * self.beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(self.docs):
            self.N += len(doc)
            if stopwords==2:
                z_n = numpy.array([0 if voca.is_stopword_id(w) else numpy.random.randint(1, self.K) for w in doc])
            else:
                z_n = numpy.random.randint(0, self.K, len(doc))
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
        return voca

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self):
        phi = self.worddist()
        log_per = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(self.docs):
            theta = self.n_m_z[m,:] / (len(doc) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
        return numpy.exp(log_per / self.N)

def lda_learning(lda, iteration):
    for i in range(iteration):
        print "-%d p=%f" % (i + 1, lda.perplexity())
        lda.inference()
    print "perplexity=%f" % lda.perplexity()

def main():
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="stopwords", type="int", help="0=exclude stop words, 1=include stop words, 2=stop words into one topic", default=1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        numpy.random.seed(options.seed)

    lda = LDA(options.K, options.alpha, options.beta)
    voca = lda.set_corpus(corpus, options.stopwords)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration)

    phi = lda.worddist()
    for k in range(options.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k,:])[:20]:
            print "%s: %f" % (voca[w], phi[k,w])

if __name__ == "__main__":
    main()
