#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation with Mixed Random Measures
#   - a parametric (grade down!?) model of DP-MRM
# (inspired via "Dirichlet Process with Mixed Random Measures : A Nonparametric Topic Model for Labeled Data"(Kim et.al, 2012))

# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
numpy.seterr(all='raise')
from scipy.special import gammaln

class LDA_MRM:
    """
        docs : [[word_id, ...], ...]
        labels : list of multiple labels of documents [[label_id, ...], ...]
        T : number of "tables" for each document (= number of topics corresponging to a document directly)
        L : number of phis(=topic-word distributions) for each label
            amount of topics = K * L,  where K : number of labels
        V : vocabulary size
    """
    def __init__(self, docs, labels, T, L, V, alpha, beta, gamma, eta):
        self.M = M = len(docs)
        self.K = K = max(max(l) for l in labels) + 1
        self.T = T
        self.L = L
        self.V = V

        self.x_ji = docs
        self.labels = labels
        """
        self.r_jk = numpy.zeros((M, K), dtype=int)
        for j, label in enumerate(labels):
            for k in label:
                self.r_jk[j, k] = 1
        """

        self.n_jt = numpy.zeros((M, T)) + alpha
        self.n_klv = numpy.zeros((K, L, V)) + beta
        self.n_kl = numpy.zeros((K, L)) + V * beta
        self.m_jk = numpy.zeros((M, K)) + eta
        self.m_kl = numpy.zeros((K, L)) + gamma # gamma_k[:, numpy.newaxis]

        self.l_jt = numpy.random.randint(L, size=(M, T))
        self.k_jt = numpy.zeros((M, T), dtype=int)
        for j in xrange(M):
            self.k_jt[j] = numpy.take(labels[j], numpy.random.randint(len(labels[j]), size=T))
            for k, l in zip(self.k_jt[j].flat, self.l_jt[j].flat):
                self.m_jk[j, k] += 1
                self.m_kl[k, l] += 1

        self.t_ji = []
        for j, x_i in enumerate(self.x_ji):
            t_i = []
            for v in x_i:
                p_t = self.n_jt[j] * self.n_klv[self.k_jt[j], self.l_jt[j], v] / self.n_kl[self.k_jt[j], self.l_jt[j]]
                t = numpy.random.multinomial(1, p_t / p_t.sum()).argmax()
                t_i.append(t)
                self.n_jt[j, t] += 1
                self.n_klv[self.k_jt[j, t], self.l_jt[j, t], v] += 1
                self.n_kl[self.k_jt[j, t], self.l_jt[j, t]] += 1
            self.t_ji.append(numpy.array(t_i))

        print self.k_jt, self.l_jt, self.m_kl

    def sampling_kl(self, j, t):
        k_old = self.k_jt[j, t]
        l_old = self.l_jt[j, t]

        print j, t, self.k_jt, self.l_jt, self.m_kl
        self.m_kl[k_old, l_old] -= 1
        self.m_jk[j, k_old] -= 1
        assert (self.m_kl >= 0).all()
        assert (self.m_jk >= 0).all()

        i_jt = numpy.flatnonzero(self.t_ji[j]==t)
        n_jt = i_jt.size
        self.n_kl[k_old, l_old] -= n_jt

        n_jtv = dict()
        for i in i_jt:
            v = self.x_ji[j][i]
            self.n_klv[k_old, l_old, v] -= 1
            if v in n_jtv:
                n_jtv[v] += 1
            else:
                n_jtv[v] = 1

        label = self.labels[j]
        n_kl = self.n_kl[label, :]
        log_f_kl = gammaln(n_kl) - gammaln(n_kl + n_jt)
        assert log_f_kl.shape == (len(label), self.L)
        for v, n in n_jtv.iteritems():
            n_klv = self.n_klv[label, :, v]
            log_f_kl += gammaln(n_klv + n) - gammaln(n_klv)
        log_p_kl = numpy.log(self.m_jk[j, label][:,numpy.newaxis] * self.m_kl[label, :]) + log_f_kl
        assert log_p_kl.shape == (len(label), self.L)
        p_kl = numpy.exp(log_p_kl.flatten() - log_p_kl.max())
        print self.m_jk, self.m_kl
        print label, log_p_kl, p_kl
        assert numpy.isfinite(log_p_kl).all()
        kl_new = numpy.random.multinomial(1, p_kl / p_kl.sum()).argmax()
        k_new = label[kl_new / self.L]
        l_new = kl_new % self.L

        self.k_jt[j, t] = k_new
        self.l_jt[j, t] = l_new

        self.m_kl[k_new, l_new] += 1
        self.m_jk[j, k_new] += 1

        self.n_kl[k_new, l_new] += n_jt
        for v, n in n_jtv.iteritems():
            self.n_klv[k_new, l_new, v] += n


    def sampling_t(self, j, i):
        v = self.x_ji[j][i]
        t_old = self.t_ji[j][i]
        k = self.k_jt[j, t_old]
        l = self.l_jt[j, t_old]
        self.n_jt[j, t_old] -= 1
        self.n_klv[k, l, v] -= 1
        self.n_kl[k, l] -= 1

        p_t = self.n_jt[j] * self.n_klv[self.k_jt[j], self.l_jt[j], v] / self.n_kl[self.k_jt[j], self.l_jt[j]]
        t_new = numpy.random.multinomial(1, p_t / p_t.sum()).argmax()

        self.t_ji[j][i] = t_new
        self.n_jt[j, t_new] += 1
        self.n_klv[self.k_jt[j, t_new], self.l_jt[j, t_new], v] += 1
        self.n_kl[self.k_jt[j, t_new], self.l_jt[j, t_new]] += 1

        """
        if t_old != t_new:
            self.m_jk[j, self.k_jt[j, t_old]] -= 1
            self.m_jk[j, self.k_jt[j, t_new]] += 1
            self.m_kl[self.k_jt[j, t_old], self.l_jt[j, t_old]] -= 1
            self.m_kl[self.k_jt[j, t_new], self.l_jt[j, t_new]] += 1
        """

    def inference(self):
        """ one iteration of inference """
        for j in xrange(self.M):
            for i in xrange(len(self.x_ji[j])):
                self.sampling_t(j, i)
            for t in xrange(self.T):
                self.sampling_kl(j, t)

    def likelihood(self):
        pass

def learning(model, iteration):
    for i in range(iteration):
        print "-%d K=%d p=%f" % (i + 1, len(model.topics), model.perplexity())
        model.inference()
    print "K=%d perplexity=%f" % (len(model.topics), model.perplexity())
    return model

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.1)
    parser.add_option("--beta", dest="beta", type="float", help="parameter of base measure H", default=0.001)
    parser.add_option("--gamma", dest="gamma", type="float", help="parameter gamma", default=0.1)
    parser.add_option("--eta", dest="eta", type="float", help="parameter eta", default=0.1)
    parser.add_option("-l", dest="L", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")
    if options.seed != None:
        numpy.random.seed(options.seed)



if __name__ == "__main__":
    main()
