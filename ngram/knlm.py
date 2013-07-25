#!/usr/bin/env python
# encode: utf-8

# n-Gram Language Model with Knerser-Ney Smoother
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import sys, codecs, re, numpy

class NGram(dict):
    def __init__(self, N, depth=1):
        self.freq = 0
        self.N = N
        self.depth = depth
    def inc(self, v):
        if self.depth <= self.N:
            if v not in self:
                self[v] = NGram(self.N, self.depth + 1)
            self[v].freq += 1
            return self[v]
    def dump(self):
        if self.depth <= self.N:
            return "%d:{%s}" % (self.freq, ",".join("'%s':%s" % (k,d.dump()) for k,d in self.iteritems()))
        return "%d" % self.freq

    def probKN(self, D, given=""):
        assert D >= 0.0 and D <= 1.0
        if given == "":
            voca = self.keys()
            n = float(self.freq)
            return voca, [self[v].freq / n for v in voca]
        else:
            if len(given) >= self.N:
                given = given[-(self.N-1):]
            voca, low_prob = self.probKN(D, given[1:])
            cur_ngram = self
            for v in given:
                if v not in cur_ngram: return voca, low_prob
                cur_ngram = cur_ngram[v]
            g = 0.0 # for normalization
            freq = []
            for v in voca:
                c = cur_ngram[v].freq if v in cur_ngram else 0
                if c > D:
                    g += D
                    c -= D
                freq.append(c)
            n = float(cur_ngram.freq)
            return voca, [(c + g * lp) / n for c, lp in zip(freq, low_prob)]

class Generator(object):
    def __init__(self, ngram):
        self.ngram = ngram
        self.start()
    def start(self):
        self.pointers = []
    def inc(self, v):
        pointers = self.pointers + [self.ngram]
        self.pointers = [d.inc(v) for d in pointers if d != None]
        self.ngram.freq += 1

def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-n", dest="ngram", type="int", help="n-gram", default=7)
    parser.add_option("-d", dest="discount", type="float", help="discount parameter of Knerser-Ney", default=0.5)
    parser.add_option("-i", dest="numgen", type="int", help="number of texts to generate", default=100)
    parser.add_option("-e", dest="encode", help="character code of input file(s)", default='utf-8')
    parser.add_option("-o", dest="output", help="output filename", default="generated.txt")
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (opt, args) = parser.parse_args()

    numpy.random.seed(opt.seed)

    START = u"\u0001"
    END = u"\u0002"

    ngram = NGram(opt.ngram)
    gen = Generator(ngram)
    for filename in args:
        with codecs.open(filename, "rb", opt.encode) as f:
            for s in f:
                s = s.strip()
                if len(s) == 0: continue
                s = START + s + END
                gen.start()
                for c in s:
                    gen.inc(c)

    D = opt.discount
    with codecs.open(opt.output, "wb", "utf-8") as f:
        for n in xrange(opt.numgen):
            st = START
            for i in xrange(1000):
                voca, prob = ngram.probKN(D, st)
                i = numpy.random.multinomial(1, prob).argmax()
                v = voca[i]
                if v == END: break
                st += v
            f.write(st[1:])
            f.write("\n")

if __name__ == "__main__":
    main()

