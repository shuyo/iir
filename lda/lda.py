#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.

from optparse import OptionParser
import sys, numpy
import vocabulary

class LDA:
    def __init__(self, K, alpha, beta):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior

    def set_corpus(self, corpus, stopwords):
        """set courpus and initialize"""
        voca = vocabulary.Vocabulary(stopwords)
        self.docs = [voca.doc_to_ids(doc) for doc in corpus]

        M = len(self.docs)
        self.V = voca.size()

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((M, self.K), dtype=int) # word count of each document and topic
        self.n_z_t = numpy.zeros((self.K, self.V), dtype=int) # word count of each topic and vocabulary
        self.n_z = numpy.zeros(self.K, dtype=int)        # word count of each topic

        for m, doc in zip(range(M), self.docs):
            z_n = numpy.random.randint(0, self.K, len(doc))
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
        return voca

    def inference(self):
        vbeta = self.V * self.beta
        for m, doc in zip(range(len(self.docs)), self.docs):
            for n, t, z in zip(range(len(doc)), doc, self.z_m_n[m]):
                # discount for n-th word t with topic z
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                denom_b = self.n_z_t.sum(axis=1) + vbeta
                p_z = (self.n_z_t[:, t] + self.beta) * (self.n_m_z[m] + self.alpha) / denom_b
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + self.V * self.beta)

    def predictive(self, doc):
        pass



def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-r", dest="reuters", help="corpus range of Reuters' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="stopwords", type="int", help="except stop words", default=1)
    (options, args) = parser.parse_args()
    if not (options.filename or options.reuters): parser.error("need corpus filename(-f) or Reuters range(-r)")

    if options.filename:
        corpus = vocabulary.load_corpus(options.filename)
    else:
        corpus = vocabulary.load_reuters(options.reuters)
        if not corpus: parser.error("Reuters range(-r) forms 'start:end'")
    lda = LDA(options.K, options.alpha, options.beta)
    voca = lda.set_corpus(corpus, options.stopwords)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    for i in range(options.iteration):
        sys.stderr.write("-%d " % (i + 1))
        lda.inference()
    #print lda.z_m_n

    phi = lda.phi()
    #for v, term in enumerate(voca):
    #    print ','.join([term]+[str(x) for x in phi[:,v]])
    for k in range(options.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k,:])[:20]:
            print "%s: %f" % (voca[w], phi[k,w])

if __name__ == "__main__":
    main()
