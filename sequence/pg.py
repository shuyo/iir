#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project Gutenberg Content Extractor with CRF

import re, glob
import pickle
from optparse import OptionParser
from crf import CRF, Features, FeatureVector


def load_dir(dir):
    '''load training/test data directory'''

    labels = []
    texts = []
    for filename in glob.glob(dir + '/*'):
        text, label = load_file(filename)
        texts.append(text)
        labels.append(label)
    return (texts, labels)

def load_file(filename):
    '''load one file of Project Gutenberg'''

    text = []
    label = []
    current_label = "H"
    f = open(filename, 'r')
    paragraph = ""
    for line in f:
        line = line.rstrip()
        if len(line)==0:
            if len(paragraph)==0:
                text[-1] += "\n"
                continue
            text.append(paragraph)
            label.append(current_label)
            paragraph = ""
            continue
        mt = re.match(r'##([A-Z]{1,3})$', line) # right tag (for training data)
        if mt:
            current_label = mt.group(1)
            continue
        paragraph += line + "\n"
    f.close()
    print filename, len(text), "paras."
    return (text, label)

def pg_features(LABELS):
    '''CRF features for Project Gutenberg Content Extractor'''

    features = Features(LABELS)
    for label in LABELS:
        # keywords
        for word in "project/gutenberg/e-?text/ebook/copyright/chapter/scanner/David Reed/encoding/contents/file/zip/web/http/email/newsletter/public domain/donation/archive/ascii/end of (the)? project gutenberg/PREFACE/INTRODUCTION/Language:/Release Date:/Character set/refund/LIMITED RIGHT".split('/'):
            features.add_feature( lambda x, y, w=word, l=label: 1 if re.search(w, x, re.I) and y == l else 0 )

        # type case
        features.add_feature( lambda x, y, l=label: 1 if x.upper() == x and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[A-Z]{3}', x) and y == l else 0 )

        # numeric
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[0-9]',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[0-9]{2}',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[0-9]{3}',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[0-9]{4}',x) and y == l else 0 )

        # line head
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'^  ', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'^    ', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'^\*', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'^\*{2}', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'^\*{3}', x, re.M) and y == l else 0 )

        # line tail
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\*$', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\*{2}$', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\*{3}$', x, re.M) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\n\n$',x) and y == l else 0 )

        # symbols
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'@',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'#',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\?',x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'\[',x) and y == l else 0 )

        # line number
        features.add_feature( lambda x, y, l=label: 1 if len(x.split("\n"))==1 and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if len(x.split("\n"))==2 and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if len(x.split("\n"))==3 and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if len(x.split("\n"))>3 and y == l else 0 )

    # labels
    for label1 in features.labels:
        features.add_feature( lambda x, y, l=label1: 1 if y == l else 0 )
        features.add_feature_edge( lambda y_, y, l=label1: 1 if y_ == l else 0 )
        for label2 in features.labels:
            features.add_feature_edge( lambda y_, y, l1=label1, l2=label2: 1 if y_ == l1 and y == l2 else 0 )

    return features

def pg_tagging(fv, text, label, crf, features, theta):
    '''tagging & output'''

    prob, ys = crf.tagging(fv, theta)
    if all(x=="H" for x in label):
        print "log_prob:", prob

        cur_text = [] # texts with current label
        cur_label = None
        for x in zip(features.id2label(ys), text):
            if cur_label != x[0]:
                pgt_output(cur_label, cur_text)
                cur_text = []
                cur_label = x[0]
            cur_text.append(x[1][0:64].replace("\n", " "))
        pgt_output(cur_label, cur_text)
    else:
        compare = zip(label, features.id2label(ys), text)
        print "log_prob:", prob, " rate:", len(filter(lambda x:x[0]==x[1], compare)), "/", len(compare)
        for x in compare:
            if x[0] != x[1]:
                print "----------", x[0], "=>", x[1]
                print x[2][0:400]

def pgt_output(label, text):
    if len(text)==0: return
    if len(text)<=7:
        for t in text: print label, t
    else:
        for t in text[:3]: print label, t
        print ": (", len(text)-6, "paragraphs)"
        for t in text[-3:]: print label, t


def main():
    parser = OptionParser()
    parser.add_option("-d", dest="training_dir", help="training data directory")
    parser.add_option("-t", dest="test_dir", help="test data directory")
    parser.add_option("-f", dest="test_file", help="test data file")
    parser.add_option("-m", dest="model", help="model file")
    parser.add_option("-l", dest="regularity", type="int", help="regularity. 0=none, 1=L1, 2=L2 [2]", default=2)
    (options, args) = parser.parse_args()
    if not options.training_dir and not options.model:
        parser.error("need training data directory(-d) or model file(-m)")

    features = pg_features(["H", "B", "F"])
    crf = CRF(features, options.regularity)
    print "features:", features.size()
    print "labels:", len(features.labels)

    if options.training_dir:
        texts, labels = load_dir(options.training_dir)
        fvs = [FeatureVector(features, x, y) for x, y in zip(texts, labels)]

        # initial parameter (pick up max in 10 random parameters)
        theta = sorted([crf.random_param() for i in range(10)], key=lambda t:crf.likelihood(fvs, t))[-1]

        # inference
        print "log likelihood (before inference):", crf.likelihood(fvs, theta)
        theta = crf.inference(fvs, theta)
        if options.model:
            f = open(options.model, 'w')
            f.write(pickle.dumps(theta))
            f.close()
    else:
        f = open(options.model, 'r')
        theta = pickle.loads(f.read())
        f.close()

    if options.test_dir:
        test_files = glob.glob(options.test_dir + '/*')
    elif options.test_file:
        test_files = [options.test_file]
    else:
        test_files = []

    i = 0
    for filename in test_files:
        print "========== test = ", i
        text, label = load_file(filename)
        pg_tagging(FeatureVector(features, text), text, label, crf, features, theta)
        i += 1

if __name__ == "__main__":
    main()

