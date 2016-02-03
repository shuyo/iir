#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Infinite Relational Model
# via 石井健一郎・上田修功 "続・わかりやすいパターン認識" Chapter 14

# This code is available under the MIT License.
# (c)2016 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
from scipy.special import betaln, gammaln

class IRM(object):
    def __init__(self, data, alpha, a, b):
        self.R = data
        self.K, self.L = data.shape
        self.alpha = alpha
        self.a = a
        self.b = b
        self.s1 = numpy.zeros(self.K, dtype=int) - 1
        self.s2 = numpy.zeros(self.L, dtype=int) - 1
        self.n1 = []
        self.n2 = []

    def update(self):
        for k in range(self.K):
            p = self.update_cluster(k, self.s1, self.s2, self.n1, self.n2, self.R)
        for l in range(self.L):
            p = self.update_cluster(l, self.s2, self.s1, self.n2, self.n1, self.R.T)

    def update_cluster(self, k, s1, s2, n1, n2, R):
        now_i = s1[k]
        s1[k] = -1
        if now_i >= 0:
            n1[now_i] -= 1
            if n1[now_i] == 0:
                n1.pop(now_i)
                s1[s1>now_i] -= 1

        c1 = len(n1)
        c2 = len(n2)
        m1, m0, m1k, m0k = count_nij(R, s1, s2, c1, c2)

        logps = numpy.zeros(c1+1)
        for i in range(c1):
            p = numpy.log(n1[i])
            p += betaln(self.a+m1[i]+m1k, self.b+m0[i]+m0k).sum()
            p -= betaln(self.a+m1[i], self.b+m0[i]).sum()
            logps[i] = p
        p = numpy.log(self.alpha)
        p += betaln(self.a+m1k, self.b+m0k).sum()
        p -= c2 * betaln(self.a, self.b)
        logps[c1] = p

        logps -= logps.max()
        ps = numpy.exp(logps)
        ps /= ps.sum()
        new_i = numpy.random.choice(c1+1, 1, p=ps)
        if new_i<c1:
            n1[new_i] += 1
        else:
            n1.append(1)
        s1[k] = new_i

    def log_posterior(self):
        log_v = log_ps(self.alpha, self.n1, self.K)
        log_v += log_ps(self.alpha, self.n2, self.L)
        m1, m0, m1k, m0k = count_nij(self.R, self.s1, self.s2, len(self.n1), len(self.n2))
        beta_ab = betaln(self.a, self.b)
        for m1i, m0i in zip(m1, m0):
            for n1, n0 in zip(m1i, m0i):
                log_v += betaln(n1 + self.a, n0 + self.b) - beta_ab
        return log_v

    def clone(self):
        import copy
        return copy.deepcopy(self)

def count_nij(R, s1, s2, c1, c2):
    m1 = numpy.zeros((c1,c2), dtype=int)    # n_(-k,+)[i,j]
    m0 = numpy.zeros((c1,c2), dtype=int)    # n~
    m1k = numpy.zeros(c2, dtype=int)        # n_(k,+)[j] where s_k=ω_i
    m0k = numpy.zeros(c2, dtype=int)        # n~
    for i, rk in zip(s1, R):
        if i>=0:
            m1i = m1[i]
            m0i = m0[i]
        else:
            m1i = m1k
            m0i = m0k
        for j, r in zip(s2, rk):
            if j<0: continue
            if r:
                m1i[j]+=1
            else:
                m0i[j]+=1
    return m1, m0, m1k, m0k

def log_ps(a, n, N):
    c = len(n)
    return c * numpy.log(a) + gammaln(n).sum() - gammaln(a + N) + gammaln(a) - gammaln(c+1)

if __name__ == "__main__":
    #R = numpy.random.randint(2, size=(25, 40))
    from numpy.random import binomial
    from numpy import concatenate as concat
    numpy.random.seed(123)
    d = 5
    phi = [[0.1, 0.7, 0.2], [0.1, 0.3, 0.9], [0.8, 0.1, 0.2]]
    orgR = concat([concat([binomial(1, p, size=(d,d)) for p in pp], axis=1) for pp in phi])
    i = numpy.arange(orgR.shape[0])
    numpy.random.shuffle(i)
    R = orgR[i,:]
    i = numpy.arange(orgR.shape[1])
    numpy.random.shuffle(i)
    R = R[:,i]

    model = IRM(R, alpha=1.0, a=1.0, b=1.0)
    maxv = -1e9
    for i in range(200):
        model.update()
        v = model.log_posterior()
        if v > maxv:
            maxv = v
            maxm = model.clone()
        print(i, v)
    RR = R[numpy.argsort(maxm.s1), :]
    RR = RR[:, numpy.argsort(maxm.s2)]
    print("--------")
    print(orgR)
    print(maxv)
    print(R)
    print(maxm.s1)
    print(maxm.s2)
    print(RR)
