#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Labeled LDA using nltk.corpus.reuters as dataset
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

import sys, string, random, numpy
from nltk.corpus import reuters
from llda import LLDA
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()
random.seed(options.seed)
numpy.random.seed(options.seed)

idlist = random.sample(reuters.fileids(), options.samplesize)

labels = []
corpus = []
for id in idlist:
    labels.append(reuters.categories(id))
    corpus.append([x.lower() for x in reuters.words(id) if x[0] in string.ascii_letters])
    reuters.words(id).close()
labelset = list(set(reduce(list.__add__, labels)))


llda = LLDA(options.K, options.alpha, options.beta)
llda.set_corpus(labelset, corpus, labels)

print "M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), options.K)

for i in range(options.iteration):
    sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
print "perplexity : %.4f" % llda.perplexity()

phi = llda.phi()
for k, label in enumerate(labelset):
    print "\n-- label %d : %s" % (k, label)
    for w in numpy.argsort(-phi[k])[:20]:
        print "%s: %.4f" % (llda.vocas[w], phi[k,w])

