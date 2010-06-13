#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Hidden Markov Model

import sys, re
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
    def __init__(self, K):
        self.K = K
        pass

    def set_corpus(self, corpus):
        self.x_ji = [] # vocabulary for each document and term
        self.vocas = []
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
            self.x_ji.append(x_i)
        self.V = len(self.vocas)

        a = [1.1] * self.K
        self.pi = dirichlet(a)
        self.A = dirichlet(a, self.K)

        # emission
        self.B = numpy.ones((self.V, self.K)) / self.V

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

        return (gamma, xi)

    def inference(self):
        pi_new = numpy.zeros(self.K)
        A_new = numpy.zeros((self.K, self.K))
        B_new = numpy.zeros((self.V, self.K))
        for x in self.x_ji:
            gamma, xi = self.Estep(x)

            # M-step
            pi_new += gamma[1]
            A_new += sum(xi) #reduce(lambda x, y: x + y, xi)
            for v, g_n in zip(x, gamma):
                B_new[v] += g_n

        self.pi = pi_new / pi_new.sum()
        self.A = A_new / (A_new.sum(1)[:, numpy.newaxis])
        self.B = B_new / B_new.sum(0)

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-k", dest="K", type="int", help="number of latent states")
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    corpus = load_corpus(options.filename)
    K = options.K or 6

    hmm = HMM(K)
    hmm.set_corpus(corpus)
    hmm.inference()


if __name__ == "__main__":
    main()
