#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Online VB Inference for HDP [Wang+ AISTATS2011]
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
from scipy.special import digamma

def golden_section_search(func, min, max):
    x1, x3 = min, max
    x2 = (x3 - x1) / (3 + math.sqrt(5)) * 2 + x1
    f1, f2, f3 = func(x1), func(x2), func(x3)
    while (x3 - x1) > 0.0001 * (max - min):
        x4 = x1 + x3 - x2
        f4 = func(x4)
        if f4 < f2:
            if x2 < x4:
                x1, x2 = x2, x4
                f1, f2 = f2, f4
            else:
                x2, x3 = x4, x2
                f2, f3 = f4, f2
        else:
            if x4 > x2:
                x3, f3 = x4, f4
            else:
                x1, f1 = x4, f4
    return x2, f2

class OnlineHDP:
    def __init__(self, K, T, alpha, beta, gamma, docs, V, kappa=0.6, tau=64):
        self.K = K
        self.T = T
        self.alpha = alpha
        self.eta = beta
        self.gamma = gamma

        self.w_jn = docs
        self.D = len(docs)
        self.V = V

        #self.lambda_kw = numpy.ones((self.K, self.V)) / self.K
        self.lambda_kw = numpy.random.gamma(0.1, 0.1, (self.K, self.V))
        self.u_k = numpy.random.gamma(1.0, 1.0, self.K - 1)
        self.v_k = numpy.random.gamma(1.0, 1.0, self.K - 1)

        self.zeta_jtn = [numpy.ones((self.T, len(w_n))) / self.T for w_n in self.w_jn]

        self.tau_t0 = tau + 1
        self.minus_kappa = -kappa

    def inference(self):
        jlist = range(self.D)
        self.phi_jkt = numpy.zeros((self.D, self.K, self.T))
        numpy.random.shuffle(jlist)
        for j in jlist:
            w_n = self.w_jn[j]

            sum_zeta = self.zeta_jtn[j].sum(axis=1)
            a_t = 1 + sum_zeta
            cum_zeta = sum_zeta.cumsum()
            # (z_1+..+z_{T-1},..,Z_{T-2}+Z_{T-1},Z_{T-1},0)
            b_t = self.alpha + cum_zeta[-1] - cum_zeta

            digamma_uk_vk = digamma(self.u_k + self.v_k)
            E_log_1_minus_beta_prime = digamma(self.v_k) - digamma_uk_vk
            E_log_beta = digamma(self.u_k) - digamma_uk_vk
            E_log_beta[1:] += E_log_1_minus_beta_prime.cumsum()[:-1]
            E_log_beta.resize(self.K) # E_log_beta[K-1] = 0

            digamma_sum_lambda = digamma(self.lambda_kw.sum(axis=1))
            E_log_p_nk = [digamma(self.lambda_kw[:,w]) - digamma_sum_lambda for w in w_n]
            log_phi_tk = numpy.dot(self.zeta_jtn[j], E_log_p_nk) + E_log_beta
            phi_kt = numpy.exp(log_phi_tk.T - log_phi_tk.max(axis=1))
            phi_kt /= phi_kt.sum(axis=0)
            self.phi_jkt[j] = phi_kt

            digamma_at_bt = digamma(a_t + b_t)
            E_log_1_minus_pi_prime = digamma(b_t) - digamma_at_bt
            E_log_pi = digamma(a_t) - digamma_at_bt
            E_log_pi[1:] += E_log_1_minus_pi_prime.cumsum()[:-1]
            E_log_pi.resize(self.T) # E_log_pi[T-1] = 0

            log_zeta_nt = numpy.dot(E_log_p_nk, phi_kt) + E_log_pi
            zeta_tn = numpy.exp(log_zeta_nt.T - log_zeta_nt.max(axis=1))
            zeta_tn /= zeta_tn.sum(axis=0)
            self.zeta_jtn[j] = zeta_tn

            rho = self.tau_t0 ** self.minus_kappa
            self.tau_t0 += 1

            partial_lambda_kw = - self.lambda_kw + self.eta
            for n, w in enumerate(w_n):
                partial_lambda_kw[:,w] += self.D * numpy.dot(phi_kt, zeta_tn[:,n])
            self.lambda_kw += rho * partial_lambda_kw

            phi_k = phi_kt.sum(axis=1)
            partial_u_k = - self.u_k + 1 + self.D * phi_k[:-1]
            self.u_k += rho * partial_u_k

            cum_phi_k = phi_k.cumsum()
            partial_v_k = - self.v_k + self.gamma + self.D * (cum_phi_k[-1] - cum_phi_k)[:-1]
            self.v_k += rho * partial_v_k

    def worddist(self):
        return self.lambda_kw / self.lambda_kw.sum(axis=1)[:, None]

    def docdist(self):
        return numpy.array([self.eachdocdist(j) for j in xrange(self.D)])

    def eachdocdist(self, j):
        theta_k = numpy.dot(self.phi_jkt[j], self.zeta_jtn[j].sum(axis=1))
        return theta_k / theta_k.sum()

    def perplexity(self, docs=None):
        if docs == None: docs = self.w_jn
        phi = self.worddist()
        log_per = 0
        N = 0
        for j, doc in enumerate(docs):
            theta = self.eachdocdist(j)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca):
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zweight = numpy.zeros(lda.K)
    wordweight = numpy.zeros((voca.size(), lda.K))
    for j, wlist in enumerate(lda.w_jn):
        phi_kn = numpy.dot(lda.phi_jkt[j], lda.zeta_jtn[j])
        zweight += phi_kn.sum(axis=1)
        for n, w in enumerate(wlist):
            wordweight[w] += phi_kn[:,n]

    phi = lda.worddist()
    for k in xrange(lda.K):
        print "\n-- topic: %d (%.2f)" % (k, zweight[k])
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %f (%.2f)" % (voca[w], phi[k,w], wordweight[w, k])

def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("--gamma", dest="gamma", type="float", help="parameter gamma", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-t", dest="T", type="int", help="number of tables for each document", default=5)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        numpy.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = OnlineHDP(options.K, options.T, options.alpha, options.beta, options.gamma, docs, voca.size())
    print "corpus=%d, words=%d, K=%d, T=%d, a=%.3f, b=%.3f, g=%.3f" % (len(corpus), len(voca.vocas), options.K, options.T, options.alpha, options.beta, options.gamma)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

if __name__ == "__main__":
    main()
