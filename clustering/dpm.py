#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dirichlet Process Mixture ( DP Gaussian Mixture Model )
# This code is available under the MIT License.
# (c)2015 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
from numpy import log
from numpy.linalg import det, slogdet
from scipy.special import gammaln


def softmax(p):
    p = numpy.exp(p - numpy.max(p))
    return p / p.sum()

def log_af(a, n):
    return gammaln(a + n) - gammaln(a)

log_pi = log(numpy.pi)


class DPM(object):
    def __init__(self, alpha, mu_0, beta, nu, S_inv):
        self.alpha = alpha
        self.mu_0 = mu_0
        self.beta = beta
        self.nu = nu
        self.S_inv = S_inv

    def setdata(self, data, sampling_init=False):
        self.data = data
        N, D = data.shape
        if sampling_init:
            self.s = numpy.zeros(N, dtype=int) - 1
            self.n = []
        else:
            self.s = numpy.zeros(N, dtype=int)
            self.n = [N]

        self.log_alpha = log(self.alpha)
        self.beta_1_beta = self.beta / (1.0 + self.beta)
        self.logdet_S_inv_nu_2 = self.nu / 2.0 * slogdet(self.S_inv)[1]
        self.log_pxk_wishart_const = D / 2.0 * (log(self.beta_1_beta) - log_pi) \
            + self.logdet_S_inv_nu_2 \
            + gammaln((self.nu + 1) / 2.0) - gammaln((self.nu + 1 - D) / 2.0)
        self.log_prod_pXi_wishart_const = D/2.0 * log(self.beta) \
            + self.logdet_S_inv_nu_2 - gammaln((self.nu - numpy.arange(D)) / 2.0).sum()
        self.log_af_alpha_N = log_af(self.alpha, N)


    def dec_n(self, k):
        j = self.s[k]
        if j >= 0:
            self.n[j] -= 1
            if self.n[j] == 0:
                self.n.pop(j)
                self.s[self.s > j] -= 1

    def inc_n(self, j_new):
        if j_new >= len(self.n):
            self.n.append(0)
        self.n[j_new] += 1

    def log_int_pxk_wishart(self, x_k):
        xkm = x_k - self.mu_0
        S_b_inv = self.S_inv + self.beta_1_beta * numpy.outer(xkm, xkm)
        return self.log_pxk_wishart_const - (self.nu + 1) / 2.0 * slogdet(S_b_inv)[1]

    def log_int_pxk_posterior(self, k, i):
        omega_i = (self.s == i)
        omega_i[k] = False

        x = self.data[omega_i]
        x_bar = x.mean(axis=0)
        x -= x_bar
        xmu = x_bar - self.mu_0

        ni, D = x.shape
        nb = ni + self.beta
        nb1nb = nb / (1 + nb)
        nn = self.nu + ni
        nn12 = (nn + 1) / 2.0

        S_q_inv = self.S_inv + numpy.dot(x.T, x) + ni * self.beta / nb * numpy.outer(xmu, xmu)

        mu_c = (ni * x_bar + self.beta * self.mu_0) / nb
        xkm = self.data[k] - mu_c
        S_r_inv = S_q_inv + nb1nb * numpy.outer(xkm, xkm)

        return D / 2.0 * (log(nb1nb) - log_pi)  \
            - nn12 * slogdet(S_r_inv)[1] + gammaln(nn12)   \
            + nn / 2.0 * slogdet(S_q_inv)[1] - gammaln((nn + 1 - D) / 2.0)

    def log_prod_pXi_wishart(self, omega_i):
        x = self.data[omega_i]
        x_bar = x.mean(axis=0)
        x -= x_bar
        xmu = x_bar - self.mu_0
        ni, D = x.shape
        nb = ni + self.beta
        nn = self.nu + ni
        S_q_inv = self.S_inv + numpy.dot(x.T, x) + ni * self.beta / nb * numpy.outer(xmu, xmu)

        return self.log_prod_pXi_wishart_const - D / 2.0 * (log(nb) + ni * log_pi) \
                   - nn / 2.0 * slogdet(S_q_inv)[1] \
                   + gammaln((nn - numpy.arange(D)) / 2.0).sum()

    def log_posterior(self):
        c = len(self.n)
        log_p = c * self.log_alpha + gammaln(self.n).sum() - self.log_af_alpha_N
        for i in xrange(c):
            log_p += self.log_prod_pXi_wishart(self.s==i)
        return log_p


    def train(self):
        for k, x in enumerate(self.data):
            self.dec_n(k)
            probs = [log(ni) + self.log_int_pxk_posterior(k, i) for i, ni in enumerate(self.n)]
            probs.append(self.log_alpha + self.log_int_pxk_wishart(x))
            self.s[k] = j = numpy.random.choice(len(probs), p=softmax(probs))
            self.inc_n(j)




def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris().data

def generate_2d_data(seed=None):
    L = numpy.array([
    [[1.7268949,-0.1640527],[-0.1640527,1.4248395],],
    [[1.68444578,0.03869354],[0.03869354,1.13205595]],
    [[1.7690384,-0.5694205],[-0.5694205,1.2501192]],
    [[3.11952232,0.02230538],[0.02230538,1.04853444]],
    [[2.0731515,-0.5526743],[-0.5526743,2.1218691]]
    ]) # ~ Wishart(15, [[0.1,0],[0,0.1]])
    M = [[-2,-4],[-3,3],[0,0],[2,4],[4,-2]]

    numpy.random.seed(seed)
    x = numpy.vstack([
        numpy.random.multivariate_normal(m, numpy.linalg.inv(2*l),100) for m,l in zip(M,L)
    ])

    #import matplotlib.pyplot as plt
    #plt.scatter(x[:,0],x[:,1])
    #plt.show()
    return x

def load_file(filename):
    import csv
    data = []
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        for x in reader:
            if all(isfloat(a) for a in x):
                data.append([float(a) for a in x])
    return numpy.array(data)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="data filename")
    parser.add_option("--iris", dest="iris", help="use iris dataset", action="store_true")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=1.0)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=1.0/3)
    parser.add_option("--nu", dest="nu", type="float", help="degree of freedom on Wishart", default=15)
    parser.add_option("--precision", dest="precision", type="float", help="precision on Wishart", default=10.0)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("--sampling", dest="sampling", help="initialize with sampling", action="store_true")
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    opt, args = parser.parse_args()

    if opt.filename:
        data = load_file(opt.filename)
    elif opt.iris:
        data = load_iris()
    else:
        data = generate_2d_data(0)

    numpy.random.seed(opt.seed)
    model = DPM(opt.alpha, data.mean(axis=0), opt.beta, opt.nu, numpy.eye(data.shape[1]) * opt.precision)
    model.setdata(data, opt.sampling)
    for epoch in xrange(opt.iteration):
        model.train()
        v = model.log_posterior()
        print(epoch, v, len(model.n), model.n)

if __name__ == "__main__":
    main()
