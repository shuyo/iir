#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project Gutenberg Content Extractor with CRF

import re, glob
#import pickle
from optparse import OptionParser
from crf import CRF, Features, FeatureVector

def load_file(filename):
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
    print filename, len(text)
    return (text, label)

def load_data(dir):
    labels = []
    texts = []
    for filename in glob.glob(dir + '/*'):
        text, label = load_file(filename)
        texts.append(text)
        labels.append(label)
    return (texts, labels)

def main():
    parser = OptionParser()
    parser.add_option("-d", dest="training_dir", help="training data directory")
    parser.add_option("-t", dest="test_dir", help="test data directory")
    parser.add_option("-l", dest="regularity", type="int", help="regularity. 0=none, 1=L1, 2=L2 [2]", default=2)
    (options, args) = parser.parse_args()
    if not options.training_dir: parser.error("need training data directory(-d)")

    LABELS = ["H", "B", "F"]
    features = Features(LABELS)
    for label in LABELS:
        # keywords
        for word in "project/gutenberg/e-?text/ebook/copyright/chapter/scanner/David Reed/encoding/contents/file/zip/web/http/email/newsletter/public domain/donation/archive/us-ascii/end of (the)? project gutenberg".split('/'):
            features.add_feature( lambda x, y, w=word, l=label: 1 if re.search(w, x, re.I) and y == l else 0 )

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
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[A-Z]{3}', x) and y == l else 0 )
        features.add_feature( lambda x, y, l=label: 1 if re.search(r'[0-9]',x) and y == l else 0 )
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
        features.add_feature( lambda x, y, l=label: 1 if y == l else 0 )
        features.add_feature_edge( lambda y_, y, l=label: 1 if y_ == l else 0 )
        features.add_feature_edge( lambda y_, y, l=label: 1 if y == y_ and y == l else 0 )

    texts, labels = load_data(options.training_dir)
    fvs = [FeatureVector(features, x, y) for x, y in zip(texts, labels)]

    print "features:", features.size()
    print "labels:", len(features.labels)

    crf = CRF(features, options.regularity)

    max_lh = -1e9
    for i in range(10):
        theta0 = crf.random_param()
        lh = crf.likelihood(fvs, theta0)
        print lh
        if max_lh < lh:
            max_lh = lh
            theta = theta0

    print "log likelihood (before inference):", crf.likelihood(fvs, theta)
    theta = crf.inference(fvs, theta)

    def tagging(fv, text, label):
        prob, ys = crf.tagging(fv, theta)
        if all(x=="H" for x in label):
            print "log_prob:", prob

            def output(label, text):
                if len(text)==0: return
                if len(text)<=7:
                    for t in text: print label, t
                else:
                    for t in text[:3]: print label, t
                    print ": (", len(text)-6, "paragraphs)"
                    for t in text[-3:]: print label, t

            cur_text = [] # texts with current label
            cur_label = None
            for x in zip(features.id2label(ys), text):
                if cur_label != x[0]:
                    output(cur_label, cur_text)
                    cur_text = []
                    cur_label = x[0]
                cur_text.append(x[1][0:64].replace("\n", " "))
            output(cur_label, cur_text)
        else:
            compare = zip(label, features.id2label(ys), text)
            print "log_prob:", prob, " rate:", len(filter(lambda x:x[0]==x[1], compare)), "/", len(compare)
            for x in compare:
                if x[0] != x[1]:
                    print "----------", x[0], "=>", x[1]
                    print x[2][0:400]

    for i in range(4):
        print "========== training = ", i
        tagging(fvs[i], texts[i], labels[i])

    if options.test_dir:
        i = 0
        for filename in glob.glob(options.test_dir + '/*'):
            print "========== test = ", i
            text, label = load_file(filename)
            tagging(FeatureVector(features, text), text, label)
            i += 1

if __name__ == "__main__":
    main()

