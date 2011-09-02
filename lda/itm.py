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

        self.n_d_k = numpy.zeros((len(docs), K)) + alpha     # word count of each document and topic
        self.n_k_w = numpy.zeros((K, V), dtype=int)
        self.n_j_k = []
        self.n_k = numpy.zeros(K) + V * beta    # word count of each topic
        self.c_j = []

        self.w_to_j = dict()

        self.z_d_n = [] # topics of words of documents
        for doc in docs:
            self.z_d_n.append( numpy.zeros(len(doc), dtype=int) - 1 )

    def get_constraint(self, words):
        if len(words) < 2:
            raise "need more than 2 words for constraint"

        constraint_id = -1
        diff_c_j = 0
        for w in words:
            if w in self.w_to_j:
                if constraint_id < 0:
                    constraint_id = self.w_to_j[w]
                elif constraint_id != self.w_to_j[w]:
                    raise "specified words have belonged into more than 2 constraints"
            else:
                diff_c_j += 1
        if diff_c_j == 0:
            raise "all specified words belonged the same constraint already"

        if constraint_id < 0:
            constraint_id = len(self.c_j)
            self.c_j.append(diff_c_j)
            self.n_j_k.append(numpy.zeros(self.K, dtype=int))
        else:
            self.c_j[constraint_id] += diff_c_j

        for w in words:
            self.w_to_j[w] = constraint_id

        return constraint_id

    def add_constraint_all(self, words):
        constraint_id = self.get_constraint(words)

        for z_d_n in self.z_d_n: z_d_n.fill(-1)
        self.n_d_k.fill(self.alpha)
        self.n_k_w.fill(0)
        for n_j_k in self.n_j_k: n_j_k.fill(0)
        self.n_k.fill(self.V * self.beta)

    def add_constraint_doc(self, words):
        constraint_id = self.get_constraint(words)

        unassigned = []
        for d, doc in enumerate(self.docs):
            if any(self.w_to_j.get(w, -1) == constraint_id for w in doc):
                for n, w in enumerate(doc):
                    k = self.z_d_n[d][n]
                    self.n_k_w[k, w] -= 1
                    self.n_k[k] -= 1
                    j = self.w_to_j.get(w, -1)
                    if j >= 0:
                        self.n_j_k[j][k] -= 1

                self.n_d_k[d].fill(self.alpha)
                self.z_d_n[d].fill(-1)
                unassigned.append(d)
        self.n_j_k[constraint_id].fill(0)
        print "unassigned all words in document [%s]" % ",".join(unassigned)

    def add_constraint_term(self, words):
        constraint_id = self.get_constraint(words)

        self.n_j_k[constraint_id].fill(0)
        for d, doc in enumerate(self.docs):
            for n, w in enumerate(doc):
                if self.w_to_j.get(w, -1) == constraint_id:
                    k = self.z_d_n[d][n]
                    self.n_d_k[d][k] -= 1
                    self.n_k_w[k, w] -= 1
                    self.n_k[k] -= 1
                    self.z_d_n[d][n] = -1

    def add_constraint_none(self, words):
        constraint_id = self.get_constraint(words)

        n_j_k = self.n_j_k[constraint_id]
        n_j_k.fill(0)
        for w in words:
            n_j_k += self.n_k_w[:, w]

    def verify_topic(self):
        n_k_w = numpy.zeros((self.K, self.V), dtype=int)
        n_j_k = numpy.zeros((len(self.c_j), self.K), dtype=int)
        for doc, z_n in zip(self.docs, self.z_d_n):
            for w, k in zip(doc, z_n):
                if k >=0:
                    n_k_w[k, w] += 1
                    j = self.w_to_j.get(w, -1)
                    if j >= 0:
                        n_j_k[j, k] += 1

        c_j = numpy.zeros(len(self.c_j), dtype=int)
        for w in self.w_to_j:
            c_j[self.w_to_j[w]] += 1

        if numpy.abs(self.n_k - self.n_k_w.sum(1) - self.V * self.beta).max() > 0.001:
            raise "there are conflicts between n_k and n_k_w"
        if numpy.abs(self.n_d_k.sum(0) - self.n_k + (self.V * self.beta - len(self.docs) * self.alpha)).max() > 0.001:

            print self.n_d_k.sum(0) - len(self.docs) * self.alpha
            print self.n_k - self.V * self.beta
            raise "there are conflicts between n_d_k and n_k"
        if numpy.any(c_j != self.c_j):
            print c_j
            print self.c_j
            raise "there are conflicts between w_to_j and c_j"
        if numpy.any(n_k_w != self.n_k_w):
            raise "there are conflicts between z_d_n and n_k_w"
        if numpy.any(n_j_k != self.n_j_k):
            raise "there are conflicts between z_d_n/w_to_j and n_j_k"

    def inference(self):
        beta = self.beta
        eta = self.eta
        for d, doc in enumerate(self.docs):
            z_n = self.z_d_n[d]
            n_d_k = self.n_d_k[d]
            for n, w in enumerate(doc):
                k = z_n[n]
                j = self.w_to_j.get(w, -1)
                if k >= 0:
                    n_d_k[k] -= 1
                    self.n_k_w[k, w] -= 1
                    self.n_k[k] -= 1
                    if j >= 0:
                        self.n_j_k[j][k] -= 1

                # sampling topic new_z for t
                if j >= 0:
                    c_j = self.c_j[j]
                    n_j_k = self.n_j_k[j]
                    p_z = n_d_k * (self.n_k_w[:, w] + eta)  * (n_j_k + c_j * beta) / ((n_j_k + c_j * eta) * self.n_k)
                else:
                    p_z = n_d_k * (self.n_k_w[:, w] + beta) / self.n_k
                new_k = z_n[n] = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                n_d_k[new_k] += 1
                self.n_k_w[new_k, w] += 1
                self.n_k[new_k] += 1
                if j >= 0:
                    self.n_j_k[j][new_k] += 1

    def worddist(self):
        """get topic-word distribution"""
        dist = (self.n_k_w + self.beta) / self.n_k[:, numpy.newaxis]
        beta = self.beta
        eta = self.eta
        for w in self.w_to_j:
            j = self.w_to_j[w]
            c_j = self.c_j[j]
            n_j_k = self.n_j_k[j]
            dist[:, w] = (self.n_k_w[:, w] + eta) * (n_j_k + c_j * beta) / ((n_j_k + c_j * eta) * self.n_k)
        return dist

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for d, doc in enumerate(docs):
            theta = self.n_d_k[d] / (len(self.docs[d]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca):
    print "\n== perplexity for each inference =="
    for i in range(iteration):
        lda.inference()
        print "-%d p=%f" % (i + 1, lda.perplexity())

    print "\n== topic-word distribution =="
    output_topic_word_dist(lda, voca)

    if len(lda.w_to_j) > 0:
        print "\n== constraints =="
        for j, w in sorted((j, w) for w, j in lda.w_to_j.items()):
            print "%d: %s [%s]" % (lda.w_to_j[w], voca.vocas[w], ",".join(str(x) for x in lda.n_k_w[:,w]))

def output_topic_word_dist(lda, voca):
    phi = lda.worddist()
    for k in range(lda.K):
        print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k])[:30]:
            print "%s: %f" % (voca[w], phi[k,w])


def main():
    import os
    import pickle
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model", help="model filename")
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-b", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.1)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.01)
    parser.add_option("--eta", dest="eta", type="float", help="parameter eta", default=100)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    parser.add_option("-c", dest="constraint", help="add constraint (wordlist which should belong to the same topic)")
    parser.add_option("-u", "--unassign", dest="unassign", help="unassign method (all/doc/term/none)", default="none")
    (options, args) = parser.parse_args()

    numpy.random.seed(options.seed)

    if options.model and os.path.exists(options.model):
        with open(options.model, "rb") as f:
            lda, voca = pickle.load(f)
    elif not (options.filename or options.corpus):
        parser.error("need corpus filename(-f) or corpus range(-b) or model(-m)")
    else:
        import vocabulary
        if options.filename:
            corpus = vocabulary.load_file(options.filename)
        else:
            corpus = vocabulary.load_corpus(options.corpus)
            if not corpus: parser.error("corpus range(-c) forms 'start:end'")
        voca = vocabulary.Vocabulary()
        docs = [voca.doc_to_ids(doc) for doc in corpus]
        if options.df > 0: docs = voca.cut_low_freq(docs, options.df)
        lda = ITM(options.K, options.alpha, options.beta, options.eta, docs, voca.size())
    param = (len(lda.docs), len(voca.vocas), options.K, options.alpha, options.beta, options.eta)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f, eta=%f" % param

    if options.constraint:
        if options.unassign == "all":
            add_constraint = lda.add_constraint_all
        elif options.unassign == "doc":
            add_constraint = lda.add_constraint_doc
        elif options.unassign == "term":
            add_constraint = lda.add_constraint_term
        elif options.unassign == "none":
            add_constraint = lda.add_constraint_none
        else:
            parser.error("unassign method(-u) must be all/doc/term/none")

        wordlist = options.constraint.split(',')
        idlist = [voca.vocas_id[w] for w in wordlist]

        print "\n== add constraint =="
        for w, v in zip(idlist, wordlist):
            print "%s [%s]" % (v, ",".join(str(x) for x in lda.n_k_w[:,w]))

        add_constraint(idlist)

        lda.verify_topic()


    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

    with open(options.model, "wb") as f:
        pickle.dump((lda, voca), f)

if __name__ == "__main__":
    main()
