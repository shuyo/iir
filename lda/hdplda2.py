#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Hierarchical Dirichlet Process - Latent Dirichlet Allocation
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# (refer to "Hierarchical Dirichlet Processes"(Teh et.al, 2005))

import numpy
from scipy.special import gammaln

class DefaultDict(dict):
    def __init__(self, v):
        self.v = v
        dict.__init__(self)
    def __getitem__(self, k):
        return dict.__getitem__(self, k) if k in self else self.v
    def update(self, d):
        dict.update(self, d)
        return self

class HDPLDA:
    def __init__(self, alpha, beta, gamma, docs, V):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.V = V
        self.M = len(docs)

        # t : table index for document j
        #     t=0 means to draw a new table
        self.using_t = [[0] for j in xrange(self.M)]

        # k : dish(topic) index
        #     k=0 means to draw a new dish
        self.using_k = [0]

        self.x_ji = docs # vocabulary for each document and term
        self.k_jt = [numpy.zeros(1 ,dtype=int) for j in xrange(self.M)]   # topics of document and table
        self.n_jt = [numpy.zeros(1 ,dtype=int) for j in xrange(self.M)]   # number of terms for each table of document
        self.n_jtv = [[None] for j in xrange(self.M)]

        self.m = 0
        self.m_k = numpy.ones(1 ,dtype=int)  # number of tables for each topic
        self.n_k = numpy.array([self.beta * self.V]) # number of terms for each topic ( + beta * V )
        self.n_kv = [DefaultDict(0)]            # number of terms for each topic and vocabulary ( + beta )

        # table for each document and term (-1 means not-assigned)
        self.t_ji = [numpy.zeros(len(x_i), dtype=int) - 1 for x_i in docs]

    def inference(self):
        for j, x_i in enumerate(self.x_ji):
            for i in xrange(len(x_i)):
                self.sampling_t(j, i)
        for j in xrange(self.M):
            for t in self.using_t[j]:
                if t != 0: self.sampling_k(j, t)

    def worddist(self):
        """return topic-word distribution without new topic"""
        return [DefaultDict(self.beta / self.n_k[k]).update(
            (v, n_kv / self.n_k[k]) for v, n_kv in self.n_kv[k].iteritems())
                for k in self.using_k if k != 0]

    def docdist(self):
        """return document-topic distribution with new topic"""

        # am_k = effect from table-dish assignment
        am_k = numpy.array(self.m_k, dtype=float)
        am_k[0] = self.gamma
        am_k *= self.alpha / am_k[self.using_k].sum()

        theta = []
        for j, n_jt in enumerate(self.n_jt):
            p_jk = am_k.copy()
            for t in self.using_t[j]:
                if t == 0: continue
                k = self.k_jt[j][t]
                p_jk[k] += n_jt[t]
            p_jk = p_jk[self.using_k]
            theta.append(p_jk / p_jk.sum())

        return numpy.array(theta)

    def perplexity(self):
        phi = [DefaultDict(1.0/self.V)] + self.worddist()
        theta = self.docdist()
        log_likelihood = 0
        N = 0
        for x_ji, p_jk in zip(self.x_ji, theta):
            for v in x_ji:
                word_prob = sum(p * p_kv[v] for p, p_kv in zip(p_jk, phi))
                log_likelihood -= numpy.log(word_prob)
            N += len(x_ji)
        return numpy.exp(log_likelihood / N)



    def dump(self, disp_x=False):
        if disp_x: print "x_ji:", self.x_ji
        print "using_t:", self.using_t
        print "t_ji:", self.t_ji
        print "using_k:", self.using_k
        print "k_jt:", self.k_jt
        print "----"
        print "n_jt:", self.n_jt
        print "n_jtv:", self.n_jtv
        print "n_k:", self.n_k
        print "n_kv:", self.n_kv
        print "m:", self.m
        print "m_k:", self.m_k
        print


    def sampling_t(self, j, i):
        """sampling t (table) from posterior"""
        self.leave_from_table(j, i)

        v = self.x_ji[j][i]
        f_k = self.calc_f_k(v)
        assert f_k[0] == 0 # f_k[0] is a dummy and will be erased

        # sampling from posterior p(t_ji=t)
        p_t = self.calc_table_posterior(j, f_k)
        if len(p_t) > 1 and p_t[1] < 0: self.dump()
        t_new = self.using_t[j][numpy.random.multinomial(1, p_t).argmax()]
        if t_new == 0:
            p_k = self.calc_dish_posterior_w(f_k)
            k_new = self.using_k[numpy.random.multinomial(1, p_k).argmax()]
            if k_new == 0:
                k_new = self.add_new_dish()
            t_new = self.add_new_table(j, k_new)

        # increase counters
        self.seat_at_table(j, i, t_new)

    def leave_from_table(self, j, i):
        t = self.t_ji[j][i]
        if t  > 0:
            k = self.k_jt[j][t]
            assert k > 0

            # decrease counters
            v = self.x_ji[j][i]
            self.n_kv[k][v] -= 1
            self.n_k[k] -= 1
            self.n_jt[j][t] -= 1
            self.n_jtv[j][t][v] -= 1

            if self.n_jt[j][t] == 0:
                self.remove_table(j, t)

    def remove_table(self, j, t):
        """remove the table where all guests are gone"""
        k = self.k_jt[j][t]
        self.using_t[j].remove(t)
        self.m_k[k] -= 1
        self.m -= 1
        assert self.m_k[k] >= 0
        if self.m_k[k] == 0:
            # remove topic (dish) where all tables are gone
            self.using_k.remove(k)

    def calc_f_k(self, v):
        return [n_kv[v] for n_kv in self.n_kv] / self.n_k

    def calc_table_posterior(self, j, f_k):
        using_t = self.using_t[j]
        p_t = self.n_jt[j][using_t] * f_k[self.k_jt[j][using_t]]
        p_x_ji = numpy.inner(self.m_k, f_k) + self.gamma / self.V
        p_t[0] = p_x_ji * self.alpha / (self.gamma + self.m)
        #print "un-normalized p_t = ", p_t
        return p_t / p_t.sum()

    def seat_at_table(self, j, i, t_new):
        assert t_new in self.using_t[j]
        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += 1

        k_new = self.k_jt[j][t_new]
        self.n_k[k_new] += 1

        v = self.x_ji[j][i]
        self.n_kv[k_new][v] += 1
        self.n_jtv[j][t_new][v] += 1

    # Assign guest x_ji to a new table and draw topic (dish) of the table
    def add_new_table(self, j, k_new):
        assert k_new in self.using_k
        for t_new, t in enumerate(self.using_t[j]):
            if t_new != t: break
        else:
            t_new = len(self.using_t[j])
            self.n_jt[j].resize(t_new+1)
            self.k_jt[j].resize(t_new+1)
            self.n_jtv[j].append(None)

        self.using_t[j].insert(t_new, t_new)
        self.n_jt[j][t_new] = 0  # to make sure
        self.n_jtv[j][t_new] = DefaultDict(0)

        self.k_jt[j][t_new] = k_new
        self.m_k[k_new] += 1
        self.m += 1

        return t_new

    def calc_dish_posterior_w(self, f_k):
        "calculate dish(topic) posterior when one word is removed"
        p_k = (self.m_k * f_k)[self.using_k]
        p_k[0] = self.gamma / self.V
        return p_k / p_k.sum()



    def sampling_k(self, j, t):
        """sampling k (dish=topic) from posterior"""
        self.leave_from_dish(j, t)

        # sampling of k
        p_k = self.calc_dish_posterior_t(j, t)
        k_new = self.using_k[numpy.random.multinomial(1, p_k).argmax()]
        if k_new == 0:
            k_new = self.add_new_dish()

        self.seat_at_dish(j, t, k_new)

    def leave_from_dish(self, j, t):
        """
        This makes the table leave from its dish and only the table counter decrease.
        The word counters (n_k and n_kv) stay.
        """
        k = self.k_jt[j][t]
        assert k > 0
        assert self.m_k[k] > 0
        self.m_k[k] -= 1
        self.m -= 1
        if self.m_k[k] == 0:
            self.using_k.remove(k)
            self.k_jt[j][t] = 0

    def calc_dish_posterior_t(self, j, t):
        "calculate dish(topic) posterior when one table is removed"
        k_old = self.k_jt[j][t]     # it may be zero (means a removed dish)
        #print "V=", self.V, "beta=", self.beta, "n_k=", self.n_k
        Vbeta = self.V * self.beta
        n_k = self.n_k.copy()
        n_jt = self.n_jt[j][t]
        n_k[k_old] -= n_jt
        n_k = n_k[self.using_k]
        log_p_k = numpy.log(self.m_k[self.using_k]) + gammaln(n_k) - gammaln(n_k + n_jt)
        log_p_k_new = numpy.log(self.gamma) + gammaln(Vbeta) - gammaln(Vbeta + n_jt)
        #print "log_p_k_new+=gammaln(",Vbeta,") - gammaln(",Vbeta + n_jt,")"

        gammaln_beta = gammaln(self.beta)
        for w, n_jtw in self.n_jtv[j][t].iteritems():
            assert n_jtw >= 0
            if n_jtw == 0: continue
            n_kw = numpy.array([n.get(w, self.beta) for n in self.n_kv])
            n_kw[k_old] -= n_jtw
            n_kw = n_kw[self.using_k]
            n_kw[0] = 1 # dummy for logarithm's warning
            if numpy.any(n_kw <= 0): print n_kw # for debug
            log_p_k += gammaln(n_kw + n_jtw) - gammaln(n_kw)
            log_p_k_new += gammaln(self.beta + n_jtw) - gammaln_beta
            #print "log_p_k_new+=gammaln(",self.beta + n_jtw,") - gammaln(",self.beta,"), w=",w
        log_p_k[0] = log_p_k_new
        #print "un-normalized p_k = ", numpy.exp(log_p_k)
        p_k = numpy.exp(log_p_k - log_p_k.max())
        return p_k / p_k.sum()

    def seat_at_dish(self, j, t, k_new):
        self.m += 1
        self.m_k[k_new] += 1

        k_old = self.k_jt[j][t]     # it may be zero (means a removed dish)
        if k_new != k_old:
            self.k_jt[j][t] = k_new

            n_jt = self.n_jt[j][t]
            if k_old != 0: self.n_k[k_old] -= n_jt
            self.n_k[k_new] += n_jt
            for v, n in self.n_jtv[j][t].iteritems():
                if k_old != 0: self.n_kv[k_old][v] -= n
                self.n_kv[k_new][v] += n


    def add_new_dish(self):
        "This is commonly used by sampling_t and sampling_k."
        for k_new, k in enumerate(self.using_k):
            if k_new != k: break
        else:
            k_new = len(self.using_k)
            if k_new >= len(self.n_kv):
                self.n_k = numpy.resize(self.n_k, k_new + 1)
                self.m_k = numpy.resize(self.m_k, k_new + 1)
                self.n_kv.append(None)
            assert k_new == self.using_k[-1] + 1
            assert k_new < len(self.n_kv)

        self.using_k.insert(k_new, k_new)
        self.n_k[k_new] = self.beta * self.V
        self.m_k[k_new] = 0
        self.n_kv[k_new] = DefaultDict(self.beta)
        return k_new



def hdplda_learning(hdplda, iteration):
    for i in range(iteration):
        hdplda.inference()
        print "-%d K=%d p=%f" % (i + 1, len(hdplda.using_k)-1, hdplda.perplexity())
    return hdplda

def output_summary(hdplda, voca, fp=None):
    if fp==None:
        import sys
        fp = sys.stdout
    K = len(hdplda.using_k) - 1
    kmap = dict((k,i-1) for i, k in enumerate(hdplda.using_k))
    dishcount = numpy.zeros(K, dtype=int)
    wordcount = [DefaultDict(0) for k in xrange(K)]
    for j, x_ji in enumerate(hdplda.x_ji):
        for v, t in zip(x_ji, hdplda.t_ji[j]):
            k = kmap[hdplda.k_jt[j][t]]
            dishcount[k] += 1
            wordcount[k][v] += 1

    phi = hdplda.worddist()
    for k, phi_k in enumerate(phi):
        fp.write("\n-- topic: %d (%d words)\n" % (hdplda.using_k[k+1], dishcount[k]))
        for w in sorted(phi_k, key=lambda w:-phi_k[w])[:20]:
            fp.write("%s: %f (%d)\n" % (voca[w], phi_k[w], wordcount[k][w]))

    fp.write("--- document-topic distribution\n")
    theta = hdplda.docdist()
    for j, theta_j in enumerate(theta):
        fp.write("%d\t%s\n" % (j, "\t".join("%.3f" % p for p in theta_j[1:])))

    fp.write("--- dishes for document\n")
    for j, using_t in enumerate(hdplda.using_t):
        fp.write("%d\t%s\n" % (j, "\t".join(str(hdplda.k_jt[j][t]) for t in using_t if t>0)))


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=numpy.random.gamma(1, 1))
    parser.add_option("--gamma", dest="gamma", type="float", help="parameter gamma", default=numpy.random.gamma(1, 1))
    parser.add_option("--beta", dest="beta", type="float", help="parameter of beta measure H", default=0.5)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("-s", dest="stopwords", type="int", help="0=exclude stop words, 1=include stop words", default=1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")
    if options.seed != None:
        numpy.random.seed(options.seed)

    import vocabulary
    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")

    voca = vocabulary.Vocabulary(options.stopwords==0)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    hdplda = HDPLDA(options.alpha, options.gamma, options.beta, docs, voca.size())
    print "corpus=%d words=%d alpha=%.3f gamma=%.3f beta=%.3f stopwords=%d" % (len(corpus), len(voca.vocas), options.alpha, options.gamma, options.beta, options.stopwords)
    #hdplda.dump()

    #import cProfile
    #cProfile.runctx('hdplda_learning(hdplda, options.iteration)', globals(), locals(), 'hdplda.profile')
    hdplda_learning(hdplda, options.iteration)
    output_summary(hdplda, voca)



if __name__ == "__main__":
    main()
