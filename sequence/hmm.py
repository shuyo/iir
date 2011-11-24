#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Hidden Markov Model

import sys, os, re, pickle
from optparse import OptionParser
#import scipy.stats
import numpy
from numpy.random import dirichlet, randn


def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:-\w+)?(?:\'\w+)?',line)
        if len(doc)>0: corpus.append([x.lower() for x in doc])
    f.close()
    return corpus

class HMM(object):
    def set_corpus(self, corpus, end_of_sentense=False):
        self.x_ji = [] # vocabulary for each document and term
        self.vocas = []
        self.vocas_id = dict()
        if end_of_sentense: self.vocas.append("(END)") # END OF SENTENCE

        for doc in corpus:
            x_i = []
            for term in doc:
                if term not in self.vocas_id:
                    voca_id = len(self.vocas)
                    self.vocas_id[term] = voca_id
                    self.vocas.append(term)
                else:
                    voca_id = self.vocas_id[term]
                x_i.append(voca_id)
            if end_of_sentense: x_i.append(0) # END OF SENTENCE
            self.x_ji.append(x_i)
        self.V = len(self.vocas)

    def init_inference(self, K, a, triangle=False):
        self.K = K

        # transition
        self.pi = dirichlet([a] * self.K) # numpy.ones(self.K) / self.K
        if triangle:
            self.A = numpy.zeros((self.K, self.K))
            for i in range(self.K):
                self.A[i, i:self.K] = dirichlet([a] * (self.K - i))
        else:
            self.A = dirichlet([a] * self.K, self.K)

        # emission
        self.B = numpy.ones((self.V, self.K)) / self.V  # numpy.tile(1.0 / self.V, (self.V, self.K))

    def save(self, file):
        numpy.savez(file + ".npz",
            x_ji = pickle.dumps(self.x_ji),
            vocas = self.vocas,
            vocas_id = pickle.dumps(self.vocas_id),
            K = self.K,
            pi = self.pi,
            A = self.A,
            B = self.B
        )

    def load(self, file):
        if not os.path.exists(file): file += ".npz"
        x = numpy.load(file)
        self.x_ji = pickle.loads(x['x_ji'])
        self.vocas = x['vocas']
        self.vocas_id = pickle.loads(x['vocas_id'])
        self.K = x['K']
        self.pi = x['pi']
        self.A = x['A']
        self.B = x['B']
        self.V = len(self.vocas)
        print self.vocas_id["html"]

    def id2words(self, x):
        return [self.vocas[v] for v in x]

    def words2id(self, x):
        return [self.vocas_id[v] for v in x]

    def dump(self):
        print "V:", self.V
        print "pi:", self.pi
        print "A:"
        for i, x in enumerate(self.A):
            print i, ":", ', '.join(["%.4f" % y for y in x])
        print "B:"
        for i, x in enumerate(self.B):
            print i, ":", ', '.join(["%.4f" % y for y in x])

    def Estep(self, x):
        N = len(x)

        alpha = numpy.zeros((N, self.K))
        c = numpy.ones(N)   # c[0] = 1
        a = self.pi * self.B[x[0]]
        alpha[0] = a / a.sum()
        for n in xrange(1, N):
            a = self.B[x[n]] * numpy.dot(alpha[n-1], self.A)
            c[n] = z = a.sum()
            alpha[n] = a / z

        beta = numpy.zeros((N, self.K))
        beta[N-1] = 1
        for n in xrange(N-1, 0, -1):
            beta[n-1] = numpy.dot(self.A, beta[n] * self.B[x[n]]) / c[n]

        likelihood = numpy.log(c).sum()
        gamma = alpha * beta

        xi_sum = numpy.outer(alpha[0], self.B[x[1]] * beta[1]) / c[1]
        for n in range(2, N):
            xi_sum += numpy.outer(alpha[n-1], self.B[x[n]] * beta[n]) / c[n]
        xi_sum *= self.A

        return (gamma, xi_sum, likelihood)

    def inference(self):
        """
        @brief one step of EM algorithm
        @return log likelihood
        """
        pi_new = numpy.zeros(self.K)
        A_new = numpy.zeros((self.K, self.K))
        B_new = numpy.zeros((self.V, self.K))
        log_likelihood = 0
        for x in self.x_ji:
            gamma, xi_sum, likelihood = self.Estep(x)
            log_likelihood += likelihood

            # M-step
            pi_new += gamma[0]
            A_new += xi_sum
            for v, g_n in zip(x, gamma):
                B_new[v] += g_n

        self.pi = pi_new / pi_new.sum()
        self.A = A_new / (A_new.sum(1)[:, numpy.newaxis])
        self.B = B_new / B_new.sum(0)

        return log_likelihood

    def sampling(self):
        z = numpy.random.multinomial(1, self.pi).argmax()
        x_n = []
        while 1:
            v = numpy.random.multinomial(1, self.B[:,z]).argmax()
            if v==0: break
            x_n.append(self.vocas[v])
            z = numpy.random.multinomial(1, self.A[z]).argmax()
        return x_n

    def Viterbi(self, x):
        N = len(x)
        w = numpy.log(self.pi) + numpy.log(self.B[x[0]])
        argmax_z_n = []
        for n in range(1, N):
            mes = numpy.log(self.A) + w[:, numpy.newaxis] # max_{z_n}( ln p(z_{n+1}|z_n) + w(z_n) )
            argmax_z_n.append(mes.argmax(0))
            w = numpy.log(self.B[x[n]]) + mes.max(0)
        z = [0] * N
        z[N-1] = w.argmax()
        for n in range(N-1, 0, -1):
            z[n-1] = argmax_z_n[n-1][z[n]]
        return z

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-k", dest="K", type="int", help="number of latent states", default=6)
    parser.add_option("-a", dest="a", type="float", help="Dirichlet parameter", default=1.0)
    parser.add_option("-i", dest="I", type="int", help="iteration count", default=10)
    parser.add_option("-t", dest="triangle", action="store_true", help="triangle")
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    corpus = load_corpus(options.filename)

    hmm = HMM()
    hmm.set_corpus(corpus, end_of_sentense=True)
    hmm.init_inference(options.K, options.a, options.triangle)
    pre_L = -1e10
    for i in range(options.I):
        log_likelihood = hmm.inference()
        print i, ":", log_likelihood
        if pre_L > log_likelihood: break
        pre_L = log_likelihood
    hmm.dump()

    for i in range(10):
        print " ".join(hmm.sampling())

    for x in corpus:
        print zip(x, hmm.Viterbi(hmm.words2id(x)))


if __name__ == "__main__":
    main()

