#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, math
from optparse import OptionParser
import scipy.stats
import scipy.special

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
        self.x_ji = [] # vocabulary for each document and term
        self.t_ji = [] # table for each document and term

        self.k_jt = [] # topic for each document and table
        self.n_jt = [] # number of terms for each document and table

        self.n_kv = [dict()] # number of terms for each topic and vocabulary
        self.n_k = [] # number of terms for each topic
        self.m_k = [] # number of tables for each topic

        self.tables = [] # available id of tables for each document
        self.topics = [0] # available id of topics

        self.vocas = []
        self.vocas_id = dict()
        self.n_terms = 0
        self.n_tables = 0

        for doc in corpus:
            x_i = []
            for term in doc:
                if term not in self.vocas_id:
                    voca_id = len(self.vocas)
                    self.vocas_id[term] = voca_id
                    self.vocas.append(term)
                    self.n_kv[0][voca_id] = 1
                else:
                    voca_id = self.vocas_id[term]
                    self.n_kv[0][voca_id] += 1
                x_i.append(voca_id)
            self.x_ji.append(x_i)

            self.k_jt.append([0])
            self.n_jt.append([len(doc)])
            self.n_terms += len(doc)
            self.t_ji.append([0] * len(doc))
            self.tables.append([0])

        self.n_k = [self.n_terms]
        self.m_k = [len(corpus)]
        self.n_tables = len(corpus)

    def dump(self, disp_x=False):
        if disp_x: print "x_ji:", self.x_ji
        print "t_ji:", self.t_ji
        print "k_jt:", self.k_jt
        print "n_kv:", self.n_kv
        print "n_jt:", self.n_jt
        print "n_k:", self.n_k
        print "m_k:", self.m_k
        print "tables:", self.tables
        print "topics:", self.topics

    def f_k_x_ji(self, k, j, i):
        v = self.x_ji[j][i]
        n_kv = self.n_kv[k][v] if v in self.n_kv[k] else 0
        return (n_kv + self.beta) / (self.n_k[k] + self.beta * len(self.vocas))

    def f_k_new_x_ji(self):
        return 1.0 / len(self.vocas)

    def log_f_k_x_jt(self, k, j, t):
        n_kv = self.n_kv[k].copy()
        n_k = self.n_k[k]
        p = 0.0
        for i in range(len(self.x_ji[j])):
            if self.t_ji[j][i] == t:
                v = self.x_ji[j][i]
                if v not in n_kv: n_kv[v] = 0
                p += math.log(n_kv[v] + self.beta) - math.log(n_k + self.beta * len(self.vocas))
                n_kv[v] += 1
                n_k += 1
        return p

    def log_f_k_new_x_jt(self, j, t):
        p = 0.0
        n_tv = dict()
        for i in range(len(self.x_ji[j])):
            if self.t_ji[j][i] == t:
                v = self.x_ji[j][i]
                if v not in n_tv: n_tv[v] = 0
                p += math.log(n_tv[v] + self.beta)
                n_tv[v] += 1

        for n in range(self.n_jt[j][t]):
            p -= math.log(n + self.beta * len(self.vocas))

        return p

    # n_?? を使わずに計算( x_ji, t_ji, k_jt のみ参照 )
    def f_k_x_ji_exactly0(self, target_k, target_j, target_i):
        target_v = self.x_ji[target_j][target_i]
        numer = self.beta
        denom = self.beta * len(self.vocas)
        for j in range(len(self.x_ji)):
            for i in range(len(self.x_ji[j])):
                if j == target_j and i == target_i: continue
                t = self.t_ji[j][i]
                k = self.k_jt[j][t]
                if k != target_k: continue
                denom += 1
                if self.x_ji[j][i] == target_v: numer += 1
        return numer / denom

    def f_k_x_ji_exactly(self, k, target_j, target_i):
        guard = lambda j,i,t:j == target_j and i == target_i
        return math.exp(self.log_p_X_k(k, guard, isdenom=True) - self.log_p_X_k(k, guard))

    # p(x_jt|X_k^{-jt})
    def log_f_k_x_jt_exactly(self, k, target_j, target_t):
        guard = lambda j,i,t:j == target_j and t == target_t
        return self.log_p_X_k(k, guard, isdenom=True) - self.log_p_X_k(k, guard)

    def log_p_X_k(self, target_k, gaurd, isdenom=False):
        n_v = [0 for v in range(len(self.vocas))]
        n = 0
        for j in range(len(self.x_ji)):
            for i in range(len(self.x_ji[j])):
                t = self.t_ji[j][i]
                k = self.k_jt[j][t]
                g = gaurd(j,i,t)
                if (k == target_k and not g) or (isdenom and g):
                    n += 1
                    n_v[self.x_ji[j][i]] += 1
        return sum([scipy.special.gammaln(self.beta + n_v[v]) for v in range(len(self.vocas))]) - scipy.special.gammaln(self.beta * len(self.vocas) + n)

    def sampling_t(self, j, i):
        assert [sum(x.values()) for x in self.n_kv] == self.n_k

        v = self.x_ji[j][i]
        t_old = self.t_ji[j][i]
        k_old = self.k_jt[j][t_old]

        self.n_kv[k_old][v] -= 1
        self.n_k[k_old] -= 1
        self.n_jt[j][t_old] -= 1

        if self.n_jt[j][t_old]==0:
            # 客がいなくなったテーブル
            self.tables[j].remove(t_old)
            self.m_k[k_old] -= 1
            self.n_tables -= 1

        if self.n_k[k_old]==0:
            # 客がいなくなった料理(トピック)
            self.topics.remove(k_old)
            self.n_kv[k_old] = dict()

        # sampling of t ( p(t_ji=t) を求める )
        p_t = []
        Z_p_t = 0
        for t in self.tables[j]:
            k_jt = self.k_jt[j][t]

            f = self.f_k_x_ji(k_jt, j, i)
            assert abs(self.f_k_x_ji_exactly(k_jt, j, i) - f) < 1e-10, self.dump()

            p = self.n_jt[j][t] * f
            p_t.append(p)
            Z_p_t += p

        p_x_ji = self.gamma / (self.n_tables + self.gamma) * self.f_k_new_x_ji()
        for k in self.topics:
            f = self.f_k_x_ji(k, j, i)
            assert abs(self.f_k_x_ji_exactly(k, j, i) - f) < 1e-10, self.dump()
            p_x_ji += self.m_k[k] / (self.n_tables + self.gamma) * f
        p_x_ji *= self.alpha
        p_t.append(p_x_ji)
        Z_p_t += p_x_ji

        p_t = [p / Z_p_t for p in p_t]
        dist = scipy.stats.rv_discrete(values=(self.tables[j] + [-1], p_t))
        t_new = dist.rvs()
        #print "p_t:", p_t, ", t_new:", t_new

        # 新しいテーブル
        if t_new < 0:
            # 空きテーブルIDを取得
            for t_new in range(len(self.n_jt[j])):
                if t_new not in self.tables[j]: break
            else:
                # 新しいテーブルID
                t_new = len(self.n_jt[j])
                self.n_jt[j].append(0)
                self.k_jt[j].append(0)
            self.tables[j].append(t_new)
            self.n_tables += 1

            # sampling of k (新しいテーブルの料理(トピック))
            p_k = []
            Z_p_k = 0
            for k in self.topics:
                f = self.f_k_x_ji(k, j, i)
                assert abs(self.f_k_x_ji_exactly(k, j, i) - f) < 1e-10, self.dump()
                p = self.m_k[k] * f
                p_k.append(p)
                Z_p_k += p
            p = self.gamma * self.f_k_new_x_ji()
            p_k.append(p)
            Z_p_k += p

            p_k = [p / Z_p_k for p in p_k]
            dist = scipy.stats.rv_discrete(values=(self.topics + [-1], p_k))
            k_new = dist.rvs()
            #print "p_k:", p_k, ", k_new:", k_new

            # 新しいトピック
            if k_new < 0:
                # 空きトピックIDを取得
                for k_new in range(len(self.n_k)):
                    if k_new not in self.topics: break
                else:
                    # 新しいテーブルID
                    k_new = len(self.n_k)
                    self.n_k.append(0)
                    self.m_k.append(0)
                    self.n_kv.append(dict())
                self.topics.append(k_new)

            self.k_jt[j][t_new] = k_new
            self.m_k[k_new] += 1
        else:
            k_new = self.k_jt[j][t_new]

        self.t_ji[j][i] = t_new

        self.n_k[k_new] += 1
        self.n_jt[j][t_new] += 1
        if v in self.n_kv[k_new]:
            self.n_kv[k_new][v] += 1
        else:
            self.n_kv[k_new][v] = 1
        assert [sum(x.values()) for x in self.n_kv] == self.n_k


    def sampling_k(self, j, t):
        k_old = self.k_jt[j][t]
        self.m_k[k_old] -= 1
        self.n_k[k_old] -= self.n_jt[j][t]
        if self.m_k[k_old] > 0:
            for i in range(len(self.x_ji[j])):
                if self.t_ji[j][i] == t:
                    v = self.x_ji[j][i]
                    assert self.n_kv[k_old][v] > 0
                    self.n_kv[k_old][v] -= 1
        else:
            assert self.n_k[k_old] == 0
            self.n_kv[k_old] = dict()
            self.topics.remove(k_old)

        # sampling of k
        # 確率が小さくなりすぎるので log で保持。最大値を引いてから正規化
        log_p_k = []
        for k in self.topics:
            f = self.log_f_k_x_jt(k, j, t)
            assert abs(f - self.log_f_k_x_jt_exactly(k, j, t)) < 1e-10, self.dump()
            log_p_k.append(f + math.log(self.m_k[k]))
        p = math.log(self.gamma) + self.log_f_k_new_x_jt(j, t)
        log_p_k.append(p)

        max_log_p_k = max(log_p_k)
        p_k = [math.exp(p - max_log_p_k) for p in log_p_k]
        Z_p_k = sum(p_k)
        p_k = [p / Z_p_k for p in p_k]

        dist = scipy.stats.rv_discrete(values=(self.topics + [-1], p_k))
        k_new = dist.rvs()
        #print "p_k:", p_k, ", k_new:", k_new

        # 新しいトピック
        if k_new < 0:
            # 空きトピックIDを取得
            for k_new in range(len(self.n_k)):
                if k_new not in self.topics: break
            else:
                # 新しいテーブルID
                k_new = len(self.n_k)
                self.n_k.append(0)
                self.m_k.append(0)
                self.n_kv.append(dict())
            self.topics.append(k_new)

        self.k_jt[j][t] = k_new
        self.m_k[k_new] += 1
        self.n_k[k_new] += self.n_jt[j][t]
        for i in range(len(self.x_ji[j])):
            if self.t_ji[j][i] == t:
                v = self.x_ji[j][i]
                if v in self.n_kv[k_new]:
                    self.n_kv[k_new][v] += 1
                else:
                    self.n_kv[k_new][v] = 1

    def inference(self):
        for j in range(len(self.x_ji)):
            for i in range(len(self.x_ji[j])):
                #print "---- j, i:", j, i
                self.sampling_t(j, i)
            for t in self.tables[j]:
                self.sampling_k(j, t)


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    corpus = load_corpus(options.filename)

    alpha = 0.5 #scipy.stats.gamma.rvs(1,scale=1)
    beta = 0.5
    gamma = 0.5 #scipy.stats.gamma.rvs(1,scale=1)
    hdplda = HDPLDA( alpha, beta, gamma )
    hdplda.set_corpus(corpus)
    hdplda.dump(True)
    for i in range(50):
        print "----", i
        hdplda.inference()
        hdplda.dump()


if __name__ == "__main__":
    main()
