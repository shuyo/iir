#!/usr/bin/env python
# -*- coding: utf-8 -*-

# [Hu, Boyd-Graber and Satinoff ACL2011] Interactive Topic Modeling
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy

class ITM:
    def __init__(self, K, alpha, beta, eta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta  = beta  # first parameter of words prior
        self.eta   = eta   # second parameter of words prior
        self.docs  = docs
        self.V = V
        self.J = 0 # constraints size

        self.n_d_k = numpy.zeros((len(docs), K)) + alpha     # word count of each document and topic
        self.n_k_w = numpy.zeros((K, V))
        self.n_j_k = []
        #self.n_j_w_k = []  # j is unique for w, then n_j_w_k == n_w_k. Therefore is this useless???
        self.n_k = numpy.zeros(K) + V * beta    # word count of each topic
        self.c_j = []

        self.w_to_j = numpy.zeros(V, dtype=int) - 1

        self.z_d_n = [] # topics of words of documents
        self.N = 0
        for doc in docs:
            N_m = len(doc)
            self.N += N_m
            self.z_d_n.append(numpy.zeros((N_m, K), dtype=int) - 1)

    def add_constraint(self, words, method="doc"):
        if len(words) < 2:
            raise "need more than 2 words for constraint"

        constraint_id = -1
        for w in words:
            if self.w_to_j[w] >= 0:
                if constraint_id < 0:
                    constraint_id = self.w_to_j[w]
                elif constraint_id != self.w_to_j[w]:
                    raise "specified words have belonged into more than 2 constraints"

        if constraint_id < 0:
            constraint_id = len(self.c_j)
            self.c_j.append(0)
            self.n_j_k.append(numpy.zeros(self.K))
            #self.n_j_w_k.append(dict())

        if method == "all":
            pass
        elif method == "doc":
            pass
        elif method == "term":
            pass
        else: # no terms are unassigned
            diff_c_j = 0
            for w in words:
                if self.w_to_j[w] == constraint_id:
                    continue
                self.w_to_j[w] = constraint_id
                self.n_j_k[constraint_id] += self.n_k_w[:, w]
                diff_c_j += 1
            if diff_c_j == 0:
                raise "all specified words belonged the same constraint already"
            self.c_j[constraint_id] += diff_c_j

    def inference(self):
        beta = self.beta
        eta = self.eta
        for d, doc in enumerate(self.docs):
            z_n = self.z_d_n[d]
            n_d_k = self.n_d_k[d]
            for n, w in enumerate(doc):
                k = z_n[n]
                j = self.w_to_j[w]
                if k >= 0:
                    n_d_k[k] -= 1
                    self.n_k_w[k, w] -= 1
                    self.n_k[k] -= 1
                    if j >= 0:
                        self.n_j_k[j][k] -= 1
                        self.n_j_k_w[j][k, w] -= 1

                # sampling topic new_z for t
                if j >= 0:
                    c_j = self.c_j[j]
                    n_j_k = self.n_j_k[j]
                    p_z = n_d_k * (self.n_k_w[:, w] + eta)  * (n_j_k + c_j * beta) / ((n_j_k + c_j * eta) * self.n_k)
                else:
                    p_z = n_d_k * (self.n_k_w[:, w] + beta) / self.n_k
                new_k = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_k
                n_d_k[new_k] += 1
                self.n_k_w[new_k, w] += 1
                self.n_k[new_k] += 1
                if j >= 0:
                    self.n_j_k[j][new_k] += 1
                    self.n_j_k_w[j][new_k, w] += 1

                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    phi = lda.worddist()
    for k in range(lda.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %f" % (voca[w], phi[k,w])

def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model", help="model filename")
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.1)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.01)
    parser.add_option("--eta", dest="eta", type="float", help="parameter eta", default=100)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    numpy.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = ITM(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

if __name__ == "__main__":
    main()
