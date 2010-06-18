#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Extract Web Content - Test
# (c)2010 Nakatani Shuyo, Cybozu Labs Inc.

import sys, os, re
from optparse import OptionParser
sys.path.append("../hmm")
from hmm import HMM

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
    parser.add_option("-t", dest="test", help="test data directory")
    parser.add_option("-m", dest="model", help="model data filename to save")
    (options, args) = parser.parse_args()
    if not options.model: parser.error("need model data filename(-m)")

    hmm = HMM()
    hmm.load(options.model)

    if options.test:
        tests = load_data(options.test)
        for x in tests:
            print zip(x, hmm.Viterbi(hmm.words2id(x)))

if __name__ == "__main__":
    main()

