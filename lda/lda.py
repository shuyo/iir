#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.

from optparse import OptionParser
import sys, re, numpy

def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line.lower())
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus

class LDA:
    def __init__(self, K, alpha, beta):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def set_corpus(self, corpus):
        """set courpus and initialize"""
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(self.docs)
        V = len(self.vocas)

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((M, self.K), dtype=int) # word count of each document and topic
        self.n_z_t = numpy.zeros((self.K, V), dtype=int) # word count of each topic and vocabulary
        self.n_z = numpy.zeros(self.K, dtype=int)        # word count of each topic

        for m, doc in zip(range(M), self.docs):
            z_n = numpy.random.randint(0, self.K, len(doc))
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self):
        V = len(self.vocas)
        for m, doc in zip(range(len(self.docs)), self.docs):
            for n, t, z in zip(range(len(doc)), doc, self.z_m_n[m]):
                # discount for n-th word t with topic z
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def predictive(self, doc):
        pass

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    corpus = load_corpus(options.filename)
    lda = LDA(options.K, options.alpha, options.beta)
    lda.set_corpus(corpus)

    for i in range(options.iteration):
        sys.stderr.write("-- %d " % (i + 1))
        lda.inference()
    #print lda.z_m_n

    phi = lda.phi()
    #for v, voca in enumerate(lda.vocas):
    #    print ','.join([voca]+[str(x) for x in phi[:,v]])
    for k in range(options.K):
        print "topic: %d" % k
        for w in numpy.argsort(-phi[k,:])[:20]:
            print "%s: %f" % (lda.vocas[w], phi[k,w])

if __name__ == "__main__":
    main()
