#!/usr/bin/env python
# encode: utf-8

# Knerser-Ney Smoother
# This code is available under the MIT License.
# (c)2012 Nakatani Shuyo / Cybozu Labs Inc.

import sys, codecs, math, re, collections

re_word = re.compile(u'[a-z\u00c0-\u024f]+')

class Distribution(dict):
    def __init__(self, arg=None):
        if arg == None:
            dict.__init__(self)
        else:
            dict.__init__(self, arg)
        self.n = self.n1 = self.n2 = self.n3 = self.n4 = 0
        self.N1 = collections.defaultdict(int)
        self.N1plus = collections.defaultdict(int)
        self.N2 = collections.defaultdict(int)
        self.N3plus = collections.defaultdict(int)
    def __setitem__(self, w, v):
        dict.__setitem__(self, w, v)
        self.n += v
        k = w[:-1]
        self.N1plus[k] += 1
        if v == 1:
            self.n1 += 1
            self.N1[k] += 1
        if v == 2:
            self.n2 += 1
            self.N2[k] += 1
        if v == 3: self.n3 += 1
        if v == 4: self.n4 += 1
        if v >= 3: self.N3plus[k] += 1
    def __getitem__(self, k):
        return dict.__getitem__(self, k) if k in self else 0

def loaddist(filename, threshold=0):
    dist = Distribution()
    with codecs.open(filename, "rb", "utf-8") as f:
        for s in f:
            w, c = s.split("\t")
            c = int(c)
            if c > threshold: dist[tuple(w.split(" "))] = c
    return dist

def maxlikelifood(gram1, gram2, gram3, text):
    w2 = w1 = ""
    for m in re_word.finditer(text):
        w = m.group(0)
        print "\np(%s) = %d / %d = %.5f" % (w, gram1[(w,)], gram1.n, float(gram1[(w,)]) / gram1.n)
        if gram1[(w1,)]>0:
            print "p(%s | %s) = %d / %d = %.5f" % (w, w1, gram2[(w1, w)], gram1[(w1,)], float(gram2[(w1, w)]) / gram1[(w1,)])
        if gram2[(w2, w1)]>0:
            print "p(%s | %s %s) = %d / %d = %.5f" % (w, w2, w1, gram3[(w2, w1, w)], gram2[(w2, w1)], float(gram3[(w2, w1, w)]) / gram2[(w2, w1)])
        w2, w1 = w1, w

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

def unigram_perplexity(gram1, test, V, alpha):
    ppl = 0.0
    N = 0
    denom = gram1.n + V * alpha
    for s in test:
        for w in s:
            p = (gram1[(w,)] + alpha) / denom
            ppl -= math.log(p)
            N += 1
    return math.exp(ppl / N)

def bigram_perplexity(gram1, gram2, test, V, alpha1, alpha2):
    ppl = 0.0
    N = 0
    for s in test:
        w1 = ""
        for w in s:
            if w1 == "":
                p = (gram1[(w,)] + alpha1) / (gram1.n + V * alpha1)
            else:
                p = (gram2[(w1, w)] + alpha2) / (gram1[(w1,)] + V * alpha2)
            ppl -= math.log(p)
            w1 = w
            N += 1
    return math.exp(ppl / N)

def trigram_perplexity(gram1, gram2, gram3, test, V, alpha1, alpha2, alpha3):
    ppl = 0.0
    N = 0
    for s in test:
        w1 = w2 = ""
        for w in s:
            if w1 == "":
                p = (gram1[(w,)] + alpha1) / (gram1.n + V * alpha1)
            elif w2 == "":
                p = (gram2[(w1, w)] + alpha2) / (gram1[(w1,)] + V * alpha2)
            else:
                p = (gram3[(w2, w1, w)] + alpha3) / (gram2[(w2, w1)] + V * alpha3)
            ppl -= math.log(p)
            w2, w1 = w1, w
            N += 1
    return math.exp(ppl / N)

def kn1_perplexity(gram1, test, V, D=None):
    if D == None:
        D = gram1.n1 / float(gram1.n1 + 2 * gram1.n2)

    ppl = 0.0
    N = 0
    for s in test:
        for w in s:
            p = (max(gram1[(w,)] - D, 0) + D * gram1.N1plus[()] / V ) / gram1.n
            ppl -= math.log(p)
            N += 1
    return math.exp(ppl / N)

def mkn_heuristic_D(gram):
    Y = gram.n1 / float(gram.n1 + 2 * gram.n2)
    D1 = 1 - 2 * Y * gram.n2 / gram.n1
    D2 = 2 - 3 * Y * gram.n3 / gram.n2
    D3 = 3 - 4 * Y * gram.n4 / gram.n3
    return (D1, D2, D3)

def mkn1_perplexity(gram1, test, V):
    D1, D2, D3 = mkn_heuristic_D(gram1)
    gamma = D1 * gram1.n1 + D2 * gram1.n2 + D3 * gram1.N3plus[()]

    ppl = 0.0
    N = 0
    for s in test:
        for w in s:
            c = gram1[(w,)]
            D = 0  if c == 0 else D1 if c == 1 else D2 if c == 2 else D3
            p = (c - D + gamma / V ) / gram1.n
            ppl -= math.log(p)
            N += 1
    return math.exp(ppl / N)

def kn2_perplexity(gram1, gram2, test, V, D1=None, D2=None):
    if D1 == None:
        D1 = gram1.n1 / float(gram1.n1 + 2 * gram1.n2)
    if D2 == None:
        D2 = gram2.n1 / float(gram2.n1 + 2 * gram2.n2)

    ppl = 0.0
    N = 0
    for s in test:
        w1 = ''
        for w in s:
            c1 = gram1[(w,)]
            p = (max(c1 - D1, 0) + D1 * gram1.N1plus[()] / V ) / gram1.n
            if (w1,) in gram1:
                c2 = gram2[(w1, w)]
                p = (max(c2 - D2, 0) + D2 * gram2.N1plus[(w1,)] * p ) / gram1[(w1,)]
            ppl -= math.log(p)
            N += 1
            w1 = w
    return math.exp(ppl / N)

def mkn2_perplexity(gram1, gram2, test, V):
    D11, D12, D13 = mkn_heuristic_D(gram1)
    D21, D22, D23 = mkn_heuristic_D(gram2)
    gamma1 = D11 * gram1.n1 + D12 * gram1.n2 + D13 * gram1.N3plus[()]

    ppl = 0.0
    N = 0
    for s in test:
        w1 = ''
        for w in s:
            c1 = gram1[(w,)]
            D = 0 if c1 == 0 else D11 if c1 == 1 else D12 if c1 == 2 else D13
            p = (c1 - D + gamma1 / V ) / gram1.n
            if (w1,) in gram1:
                c2 = gram2[(w1, w)]
                D = 0 if c2 == 0 else D21 if c2 == 1 else D22 if c2 == 2 else D23
                gamma = D21 * gram2.N1[(w1,)] + D22 * gram2.N2[(w1,)] + D23 * gram2.N3plus[(w1,)]
                p = (c2 - D + gamma * p ) / gram1[(w1,)]
            ppl -= math.log(p)
            N += 1
            w1 = w
    return math.exp(ppl / N)

def kn3_perplexity(gram1, gram2, gram3, test, V, D1=None, D2=None, D3=None):
    if D1 == None:
        D1 = gram1.n1 / float(gram1.n1 + 2 * gram1.n2)
    if D2 == None:
        D2 = gram2.n1 / float(gram2.n1 + 2 * gram2.n2)
    if D3 == None:
        D3 = gram3.n1 / float(gram3.n1 + 2 * gram3.n2)

    ppl = 0.0
    N = 0
    for s in test:
        w1 = w2 = ''
        for w in s:
            c1 = gram1[(w,)]
            p = (max(c1 - D1, 0) + D1 * gram1.N1plus[()] / V ) / gram1.n
            if (w1,) in gram1:
                c2 = gram2[(w1, w)]
                p = (max(c2 - D2, 0) + D2 * gram2.N1plus[(w1,)] * p ) / gram1[(w1,)]
                if (w2, w1) in gram2:
                    c3 = gram3[(w2, w1, w)]
                    p = (max(c3 - D3, 0) + D3 * gram3.N1plus[(w2, w1,)] * p ) / gram2[(w2, w1)]
            ppl -= math.log(p)
            N += 1
            w2, w1 = w1, w
    return math.exp(ppl / N)

def mkn3_perplexity(gram1, gram2, gram3, test, V):
    D11, D12, D13 = mkn_heuristic_D(gram1)
    D21, D22, D23 = mkn_heuristic_D(gram2)
    D31, D32, D33 = mkn_heuristic_D(gram3)
    gamma1 = D11 * gram1.n1 + D12 * gram1.n2 + D13 * gram1.N3plus[()]

    ppl = 0.0
    N = 0
    for s in test:
        w2 = w1 = ''
        for w in s:
            c1 = gram1[(w,)]
            D = 0 if c1 == 0 else D11 if c1 == 1 else D12 if c1 == 2 else D13
            p = (c1 - D + gamma1 / V ) / gram1.n
            if (w1,) in gram1:
                c2 = gram2[(w1, w)]
                D = 0 if c2 == 0 else D21 if c2 == 1 else D22 if c2 == 2 else D23
                gamma = D21 * gram2.N1[(w1,)] + D22 * gram2.N2[(w1,)] + D23 * gram2.N3plus[(w1,)]
                p = (c2 - D + gamma * p ) / gram1[(w1,)]
                if (w2, w1) in gram2:
                    c3 = gram3[(w2, w1, w)]
                    D = 0 if c3 == 0 else D31 if c3 == 1 else D32 if c3 == 2 else D33
                    gamma = D31 * gram3.N1[(w2,w1)] + D32 * gram3.N2[(w2,w1)] + D33 * gram3.N3plus[(w2,w1)]
                    p = (c3 - D + gamma * p ) / gram2[(w2, w1)]
            ppl -= math.log(p)
            N += 1
            w2, w1 = w1, w
    return math.exp(ppl / N)

def main():
    import optparse, random, nltk

    parser = optparse.OptionParser()
    parser.add_option("-c", dest="corpus", help="corpus module name under nltk.corpus (e.g. brown, reuters)", default='brown')
    parser.add_option("-r", dest="testrate", type="float", help="rate of test dataset in corpus", default=0.1)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()

    random.seed(opt.seed)

    m = __import__('nltk.corpus', globals(), locals(), [opt.corpus], -1)
    corpus = getattr(m, opt.corpus)
    ids = corpus.fileids()
    D = len(ids)
    print "found corpus : %s (D=%d)" % (opt.corpus, D)

    testids = set(random.sample(ids, int(D * opt.testrate)))
    trainids = [id for id in ids if id not in testids]
    trainwords = [w.lower() for w in corpus.words(trainids)]

    freq1 = nltk.FreqDist(trainwords)
    gram1 = Distribution()
    for w, c in freq1.iteritems():
        gram1[(w,)] = c
    print "# of terms=%d, vocabulary size=%d" % (gram1.n, len(gram1))

    gram2 = Distribution()
    for w, c in nltk.FreqDist(nltk.bigrams(trainwords)).iteritems():
        gram2[w] = c
    gram3 = Distribution()
    for w, c in nltk.FreqDist(nltk.trigrams(trainwords)).iteritems():
        gram3[w] = c

    #maxlikelifood(gram1, gram2, gram3, "this is a pen")
    #maxlikelifood(gram1, gram2, gram3, "the associated press microsoft looking at making its own smartphone?")

    testset = []
    voca = set(freq1.iterkeys())
    for id in testids:
        f = corpus.words(id)
        doc = [w.lower() for w in f]
        f.close()

        testset.append(doc)
        for w in doc:
            voca.add(w)
    V = len(voca)

    D1 = gram1.n1 / float(gram1.n1 + 2 * gram1.n2)
    D2 = gram2.n1 / float(gram2.n1 + 2 * gram2.n2)
    D3 = gram3.n1 / float(gram3.n1 + 2 * gram3.n2)

    print "\nUNIGRAM:"
    alpha1, minppl = golden_section_search(lambda a:unigram_perplexity(gram1, testset, V, a), 0.0001, 1.0)
    print "additive smoother: alpha1=%.4f, perplexity=%.3f" % (alpha1, minppl)
    print "Kneser-Ney: heuristic D=%.3f, perplexity=%.3f" % (D1, kn1_perplexity(gram1, testset, V, D1))
    D1min, minppl = golden_section_search(lambda d:kn1_perplexity(gram1, testset, V, d), 0.0001, 0.9999)
    print "Kneser-Ney: minimum D=%.3f, perplexity=%.3f" % (D1min, minppl)
    print "modified Kneser-Ney: perplexity=%.3f" % mkn1_perplexity(gram1, testset, V)

    print "\nBIGRAM:"
    alpha2, minppl = golden_section_search(lambda a:bigram_perplexity(gram1, gram2, testset, V, alpha1, a), 0.0001, 1.0)
    print "additive smoother: alpha2=%.4f, perplexity=%.3f" % (alpha2, minppl)
    print "Kneser-Ney: heuristic D=%.3f, perplexity=%.3f" % (D2, kn2_perplexity(gram1, gram2, testset, V, D1, D2))
    D2min, minppl = golden_section_search(lambda a:kn2_perplexity(gram1, gram2, testset, V, D1, a), 0.0001, 0.9999)
    print "Kneser-Ney: minimum D=%.3f, perplexity=%.3f" % (D2min, minppl)
    print "modified Kneser-Ney: perplexity=%.3f" % mkn2_perplexity(gram1, gram2, testset, V)

    print "\nTRIGRAM:"
    alpha3, minppl = golden_section_search(lambda a:trigram_perplexity(gram1, gram2, gram3, testset, V, alpha1, alpha2, a), 0.0001, 1.0)
    print "additive smoother: alpha3=%.4f, perplexity=%.3f" % (alpha3, minppl)
    print "Kneser-Ney: heuristic D=%.3f, perplexity=%.3f" % (D3, kn3_perplexity(gram1, gram2, gram3, testset, V, D1, D2, D3))
    D3min, minppl = golden_section_search(lambda a:kn3_perplexity(gram1, gram2, gram3, testset, V, D1, D2, a), 0.0001, 0.9999)
    print "Kneser-Ney: minimum D=%.3f, perplexity=%.3f" % (D3min, minppl)
    print "modified Kneser-Ney: perplexity=%.3f" % mkn3_perplexity(gram1, gram2, gram3, testset, V)

if __name__ == "__main__":
    main()
