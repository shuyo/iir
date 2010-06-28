#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project Gutenberg Content Extractor with CRF

import re
#import pickle
from optparse import OptionParser
from crf import CRF, Features, FeatureVector

def load_text(filename, limit=1e9):
    labels = []
    text = []
    current_label = "H"
    f = open(filename, 'r')
    for line in f:
        m = re.match(r'##([A-Z]{1,3})$', line)
        if m:
            current_label = m.group(1)
            continue
        text.append(line)
        labels.append(current_label)
        if len(text) >= limit: break
    f.close()
    return (text, labels)

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="text filename")
    parser.add_option("-l", dest="regularity", type="int", help="regularity. 0=none, 1=L1, 2=L2 [2]", default=2)
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    text, labels = load_text(options.filename)

    features = Features(labels)
    for label in features.labels:
        for keyword in "project gutenberg etext copyright chapter".split():
            features.add_feature( lambda x, y: 1 if re.search(keyword, x, re.I) and y == label else 0 )
        features.add_feature( lambda x, y: 1 if y == label else 0 )
        features.add_feature_edge( lambda y_, y: 1 if y_ == label else 0 )

    features.add_feature( lambda x, y: 1 if re.search(r"[A-Z]{3}", x) else 0 )
    features.add_feature( lambda x, y: 1 if re.search(r"[0-9]", x) else 0 )
    features.add_feature( lambda x, y: 1 if x.find("@")>=0 else 0 )
    features.add_feature( lambda x, y: 1 if x.find("#")>=0 else 0 )
    features.add_feature( lambda x, y: 1 if x.startswith("*") else 0 )
    features.add_feature( lambda x, y: 1 if x.startswith("  ") else 0 )
    features.add_feature( lambda x, y: 1 if x.startswith("    ") else 0 )
    features.add_feature( lambda x, y: 1 if x.startswith("        ") else 0 )
    features.add_feature( lambda x, y: 1 if len(x.strip()) == 0 else 0 )

    #features.add_feature_edge( lambda y_, y: 1 if len(x_.strip()) == 0 else 0 )

    fv = FeatureVector(features, text, labels)

    crf = CRF(features, options.regularity)
    theta = crf.random_param()

    print "theta:", theta
    print "log likelihood:", crf.likelihood(fv, theta)
    print "tagging:", crf.tagging(fv, theta)

    theta = crf.inference(fv, theta)

    print "theta:", theta
    print "log likelihood:", crf.likelihood(fv, theta)
    print "tagging:", crf.tagging(fv, theta)


if __name__ == "__main__":
    main()

