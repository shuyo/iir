#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Extract Web Content with HMM
# (c)2010 Nakatani Shuyo, Cybozu Labs Inc.

import sys, os, re
from optparse import OptionParser
sys.path.append("../hmm")
from hmm import HMM
#import numpy
#from numpy.random import dirichlet, randn

def load_data(directory):
    import glob
    htmllist = glob.glob(os.path.join(directory, "*.html"))
    features = []
    for filename in htmllist:
        taglist = []
        f = open(filename, 'r')
        for line in f:
            tags = re.findall(r'<(\w+)',line)
            if len(tags)>0: taglist.extend([x.lower() for x in tags])
        f.close()
        features.append(taglist)
    return features

def main():
    parser = OptionParser()
    parser.add_option("-d", dest="training", help="training data directory")
    parser.add_option("-k", dest="K", type="int", help="number of latent states", default=6)
    parser.add_option("-a", dest="a", type="float", help="Dirichlet parameter", default=1.0)
    parser.add_option("-i", dest="I", type="int", help="iteration count", default=10)
    parser.add_option("-m", dest="model", help="model data filename to save")
    (options, args) = parser.parse_args()
    if not options.training: parser.error("need training data directory(-d)")

    features = load_data(options.training)

    hmm = HMM()
    hmm.set_corpus(features)
    hmm.init_inference(options.K, options.a)
    pre_L = -1e10
    for i in range(options.I):
        log_likelihood = hmm.inference()
        print i, ":", log_likelihood
        if pre_L > log_likelihood: break
        pre_L = log_likelihood
    if options.model:
        hmm.save(options.model)
    else:
        hmm.dump()

if __name__ == "__main__":
    main()

