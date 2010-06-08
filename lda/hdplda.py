#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re
from optparse import OptionParser
import scipy.stats

"""
import numpy
import scipy.special
import matplotlib.pyplot
"""

def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0: corpus.append(doc)
    f.close()
    return corpus

class HDPLDA:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_corpus(self, corpus):
        self.t_ji = [] # table for each document and term
        self.k_jt = [] # topic for each document and table

        self.n_kv = [dict()] # number of terms for each topic and vocabulary
        self.n_jt = [] # number of terms for each document and table
        self.n_k = [] # number of terms for each topic

        self.m_k = [] # number of tables for each topic
        self.m_j = [] # number of tables for each document

        self.tables = [] # available id of tables
        self.topics = [0] # available id of topics

        self.vocas = []
        self.vocas_id = dict()
        self.n_terms = 0
        self.n_tables = 0

        self.x_ji = []

        for doc in corpus:
            x_i = []
            for term in doc:
                if term not in self.vocas_id:
                    voca_id = len(self.vocas)
                    self.vocas_id[term] = voca_id
                    self.vocas.append(term)
                    self.n_kv[0][voca_id] = 0
                else:
                    voca_id = self.vocas_id[term]
                    self.n_kv[0][voca_id] += 1
                 x_i.append(voca_id)
            self.x_ji.append(x_i)

            self.k_jt.append([0])
            self.n_jt.append([len(doc)])
            self.n_terms += len(doc)
            self.m_j.append(1)
            self.t_ji.append([0] * len(doc))
            self.tables.append([0])

        self.n_k = [self.n_terms]
        self.m_k = [len(corpus)]
        self.n_tables = len(corpus)

    def f_k_x_ji(self, k, j, i):
        v = self.x_ji[j][i]
        return (self.n_kv[k][v] + self.beta) / (self.n_k[k] + self.beta * len(self.vocas)

    def f_k_new_x_ji(self):
        return 1.0 / len(self.vocas)

    def f_k_x_jt(self, k, j, t):
        t_i = self.t_ji[j]
        x_jt = [i for i in range(len(t_i)) if t_i[i]==t]
        f = 1.0
        for i in x_jt:
            f *= self.f_k_x_ji(k, j, i)
        return f

    def f_k_new_x_jt(self, k, j, t):
        # TODO
        pass

    def inference(self):
        # TODO
        pass

    def sampling_t(self, j, i):
        v = self.x_ji[j][i]
        t_old = self.t_ji[j][i]
        k_old = self.k_jt[j][t_old]

        self.n_kv[k_old][v] -= 1
        self.n_k[k_old] -= 1
        self.n_jt[j][t_old] -= 1

        if self.n_jt[j][t_old]==0:
            # TODO: 客がいなくなったテーブル
            pass

        if self.n_k[k_old]==0:
            # TODO: 客がいなくなった料理(トピック)
            pass

        # p(t_ji=t) を求める
        probs = []
        for t in self.tables[j]:
            k_jt = self.k_jt[j][t]
            p = n_jt[j][t] * self.f_k_x_ji(k_jt, j, i)
            probs.append(p)

        p_x_ji = self.gamma / (self.n_tables + self.gamma) * self.f_k_new_x_ji()
        for k in self.topics:
            p_x_ji += self.m_k[k] / (self.n_tables + self.gamma) * self.f_k_x_ji(k, j, i)
        probs.append(self.alpha * p_x_ji)

        # サンプリング
        dist = scipy.stats.rv_discrete(values=(self.tables[j] + [-1], probs))
        t_new = dist.rcv()
        if t_new < 0:
            # TODO: 新しいテーブル
            pass
        else:
            k_new = self.k_jt[j][t_new]

        self.n_kv[k_new][v] += 1
        self.n_k[k_new] += 1
        self.n_jt[j][t_new] += 1

    def sampling_k(self, j, t):
        k = k_jt[j][t]
        # TODO
        pass

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    corpus = load_corpus(options.filename)

    alpha = scipy.stats.gamma.rvs(1,scale=1)
    beta = 0.5
    gamma = scipy.stats.gamma.rvs(1,scale=1)
    hdplda = HDPLDA( alpha, beta, gamma )
    hdplda.set_corpus(corpus)




if __name__ == "__main__":
    main()

