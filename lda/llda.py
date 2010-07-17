#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Labeled Latent Dirichlet Allocation
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.
# refer to Ramage+, Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora(EMNLP2009)

from optparse import OptionParser
import re, numpy

def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0: corpus.append(doc)
    f.close()
    return corpus

class LLDA:
    def __init__(self, K, alpha, beta):
        self.K = K
        self.alpha = alpha
        self.beta = beta

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def set_corpus(self, corpus):
        self.vocas = []
        self.vocas_id = dict()
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]
        M = len(corpus)
        V = len(self.vocas)

        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=int)
        self.n_z_t = numpy.zeros((self.K, V), dtype=int)
        self.n_z = numpy.zeros(self.K, dtype=int)

        for m, doc in enumerate(self.docs):
            N_m = len(doc)
            z_n = numpy.random.randint(self.K, size=N_m)
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self):
        V = len(self.vocas)
        for m, doc in enumerate(self.docs):
            for n in range(len(doc)):
                t = doc[n]
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def perplexity(self, doc):
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

    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(corpus)

    for i in range(options.iteration):
        print "----", i + 1
        llda.inference()
        print llda.z_m_n


if __name__ == "__main__":
    main()
