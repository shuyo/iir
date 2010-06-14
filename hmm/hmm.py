#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Hidden Markov Model

import sys, re, math
from optparse import OptionParser
#import scipy.stats
import numpy
from numpy.random import dirichlet, randn


def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:-\w+)?(?:\'\w+)?',line)
        if len(doc)>0: corpus.append(doc)
    f.close()
    return corpus

class HMM:
    def __init__(self, K, a, triangle=False):
        self.K = K

        # transition
        self.pi = dirichlet([a] * self.K) # numpy.ones(self.K) / self.K
        if triangle:
            self.A = numpy.zeros((self.K, self.K))
            for i in range(self.K):
                self.A[i, i:self.K] = dirichlet([a] * (self.K - i))
        else:
            self.A = dirichlet([a] * self.K, self.K)

    def set_corpus(self, corpus):
        self.x_ji = [] # vocabulary for each document and term
        self.vocas = ["(END)"]
        self.vocas_id = dict()

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
            x_i.append(0) # END MARK
            self.x_ji.append(x_i)
        self.V = len(self.vocas)

        # emission
        self.B = numpy.ones((self.V, self.K)) / self.V

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

        alpha = [None] * N
        c = numpy.ones(N)
        alpha[0] = self.pi * self.B[x[0]]
        for n in range(1, N):
            a = self.B[x[n]] * numpy.dot(alpha[n-1], self.A)
            z = a.sum()
            alpha[n] = a / z
            c[n] = z
        
        beta = [None] * N
        beta[N-1] = numpy.ones(self.K)
        for n in range(N-1, 0, -1):
            beta[n-1] = numpy.dot(self.A, beta[n] * self.B[x[n]]) / c[n]

        likelihood = numpy.prod(c)
        gamma = [a * b for a, b in zip(alpha, beta)]
        xi = [ self.A * numpy.dot(alpha[n-1].T, self.B[x[n]] * beta[n]) / c[n] for n in range(1, N)]

        return (gamma, xi, likelihood)

    def inference(self):
        pi_new = numpy.zeros(self.K)
        A_new = numpy.zeros((self.K, self.K))
        B_new = numpy.zeros((self.V, self.K))
        log_likelihood = 0
        for x in self.x_ji:
            gamma, xi, likelihood = self.Estep(x)
            log_likelihood += math.log(likelihood)

            # M-step
            pi_new += gamma[1]
            A_new += sum(xi) #reduce(lambda x, y: x + y, xi)
            for v, g_n in zip(x, gamma):
                B_new[v] += g_n

        self.pi = pi_new / pi_new.sum()
        self.A = A_new / (A_new.sum(1)[:, numpy.newaxis])
        self.B = B_new / B_new.sum(0)

        return log_likelihood

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

    hmm = HMM(options.K, options.a, options.triangle)
    hmm.set_corpus(corpus)
    for i in range(options.I):
        log_likelihood = hmm.inference()
        print i, ":", log_likelihood
    hmm.dump()


if __name__ == "__main__":
    main()

