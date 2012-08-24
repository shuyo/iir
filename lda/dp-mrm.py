#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dirichlet Process with Mixed Random Measures
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.
# (refer to "Dirichlet Process with Mixed Random Measures : A Nonparametric Topic Model for Labeled Data"(Kim et.al, 2012))

import numpy
numpy.seterr(all='raise')
from scipy.special import gammaln

class DP_MRM:
    def __init__(self, docs, labels, V, alpha, beta, gamma, eta):
        """
            docs : [[word_id, ...], ...]
            labels : list of multiple labels of documents [[label_id, ...], ...]
            T : number of "tables" for each document (= number of topics corresponging to a document directly)
            L : number of dishes(=topic-word distributions) for each label
                amount of topics = K * L,  where K is number of labels
            V : vocabulary size
        """

        self.M = M = len(docs)
        self.K = K = max(max(l) for l in labels) + 1
        self.V = V

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        # t is table index for document j, t=0 means a new table
        # l is dish index for label k, l=0 means anew dish
        # v is word id and very sparse, v = 0, ..., V-1
        # j is document index, j = 0, ..., M-1
        # k is label index, k = 0, ..., K-1

        self.using_t = [numpy.array([0], dtype=int) for j in xrange(M)]
        self.using_l = [numpy.array([0], dtype=int) for j in xrange(K)]

        self.n_jt = [numpy.array([0], dtype=int) for j in xrange(M)]    # + 0
        self.n_klv = [[dict()] for j in xrange(K)]                      # + beta
        self.n_kl = [numpy.array([V * beta]) for j in xrange(K)]        # + V*beta

        self.m_jk = numpy.zeros((M, K)) + eta                           # + eta
        self.m_kl = [numpy.array([0], dtype=int) for j in xrange(K)]    # + 0

        self.kl_jt = [numpy.array([[0,0]],dtype=int) for j in xrange(M)]

    def sampling_t(self, j, i):
        v = self.x_ji[j][i]
        t_old = self.t_ji[j][i]
        assert t_old > 0
        assert t_old in self.using_t[j]

        k, l = self.kl_jt[j][t_old]
        assert l > 0
        assert l in self.using_l[k]

        self.n_jt[j][t_old] -= 1
        self.n_klv[k][l][v] -= 1
        self.n_kl[k][l] -= 1
        assert self.n_jt[j][t_old] >= 0
        assert self.n_klv[k][l][v] >= 0
        assert self.n_kl[k][l] >= 0

        if self.n_jt[j][t_old] == 0:
            self.remove_table(j, t_old)

        kllist = self.kl_jt[j][self.using_t[j]]
        p_t = self.n_jt[j][self.using_t[j]] * [self.n_klv[k][l][v] / self.n_kl[k][l] for k, l in kllist]
        p_t[0] = self.alpha * self.wordprob_on_new_table(v)     # probability for a new table
        t_new = self.using_t[j][numpy.random.multinomial(1, p_t / p_t.sum()).argmax()]

        if t_new == 0:
            t_new = self.insert_table(j)
            assert t_new > 0

        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += 1

        k,l = self.kl_jt[j][t_new]
        if v in self.n_klv[k][l]:
            self.n_klv[k][l][v] += 1
        else:
            self.n_klv[k][l] = {v: self.beta + 1}
        self.n_kl[k][l] += 1

    def wordprob_on_new_table(self, j, v):
        prob = 0
        for k in xrange(self.K):
            pk = self.gamma / self.V  # gamma * f_kl_new
            for l in self.using_l[k]:
                if l > 0:
                    pk += self.m_kl[k][l] * self.n_klv[k][l][v] / self.n_kl[k][l]
            prob += self.m_jk[j, k] * pk / (self.m_kl[k].sum() + self.gamma)
        return prob / (self.m_jk[j].sum() + self.K * self.eta)

    def insert_table(self, j):
        """新しいテーブルを追加する。追加されたテーブルidを返す"""
        if self.using_t[j][-1] + 1 == len(self.using_t[j]):
            # expand tables and use the last one
            t_new = len(self.using_t[j])
            self.using_t[j].append(t_new)
            self.n_jt[j].resize(t_new + 1)
            self.kl_jt[j].resize(t_new + 1)
        else:
            # reuse the removed table
            for t_new, t in enumerate(self.using_t[j]):
                if t_new != t: break
            self.using_t[j].insert(t_new, t_new)

        self.n_jt[j][t_new] = 0  # to make sure
        self.kl_jt[j][t_new] = self.sampling_kl_for_new_table(j)

    def sampling_kl_for_new_table(self, j):
        pass

    def remove_table(self, j, t):
        """着席者がいなくなりテーブルが無くなるときの処理"""
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
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=numpy.random.gamma(1, 1))
    parser.add_option("--gamma", dest="gamma", type="float", help="parameter gamma", default=numpy.random.gamma(1, 1))
    parser.add_option("--base", dest="base", type="float", help="parameter of base measure H", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="initial number of topics", default=1)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("-s", dest="stopwords", type="int", help="0=exclude stop words, 1=include stop words", default=1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")
    if options.seed != None:
        numpy.random.seed(options.seed)



if __name__ == "__main__":
    main()
