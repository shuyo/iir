#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project Gutenberg Content Extractor with CRF

import numpy
import time
from optparse import OptionParser
from crf import CRF, Features, FeatureVector


def main():
    def load_data(data):
        texts = []
        labels = []
        text = []
        data = "\n" + data + "\n"
        for line in data.split("\n"):
            line = line.strip()
            if len(line) == 0:
                if len(text)>0:
                    texts.append(text)
                    labels.append(label)
                text = []
                label = []
            else:
                token, info, chunk = line.split()
                text.append((token, info))
                label.append(chunk)
        return (texts, labels)

    texts, labels = load_data("""
    This DT B-NP
    temblor-prone JJ I-NP
    city NN I-NP
    dispatched VBD B-VP
    inspectors NNS B-NP
    , , O

    firefighters NNS B-NP
    and CC O
    other JJ B-NP
    earthquake-trained JJ I-NP
    personnel NNS I-NP
    to TO B-VP
    aid VB I-VP
    San NNP B-NP
    Francisco NNP I-NP
    . . O
    """)

    print texts, labels

    test_texts, test_labels = load_data("""
    Rockwell NNP B-NP
    said VBD B-VP
    the DT B-NP
    agreement NN I-NP
    calls VBZ B-VP
    for IN B-SBAR
    it PRP B-NP
    to TO B-VP
    supply VB I-VP
    200 CD B-NP
    additional JJ I-NP
    so-called JJ I-NP
    shipsets NNS I-NP
    for IN B-PP
    the DT B-NP
    planes NNS I-NP
    . . O
    """)

    features = Features(labels)
    tokens = dict([(i[0],1) for x in texts for i in x]).keys()
    infos = dict([(i[1],1) for x in texts for i in x]).keys()

    for label in features.labels:
        for token in tokens:
            features.add_feature( lambda x, y, l=label, t=token: 1 if y==l and x[0]==t else 0 )
        for info in infos:
            features.add_feature( lambda x, y, l=label, i=info: 1 if y==l and x[1]==i else 0 )
    features.add_feature_edge( lambda y_, y: 0 )

    fvs = [FeatureVector(features, x, y) for x, y in zip(texts, labels)]
    fv = fvs[0]
    text_fv = FeatureVector(features, test_texts[0]) # text sequence without labels


    crf = CRF(features, 0)
    theta0 = crf.random_param()
    print "initial log likelihood:", crf.likelihood(fvs, theta0)


    print ">> Steepest Descent"
    theta = theta0.copy()
    eta = 0.5
    t = time.time()
    for i in range(20):
        theta += eta * crf.gradient_likelihood(fvs, theta)
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.95
    print "time = %.3f, relevant features = %d / %d" % (time.time() - t, (numpy.abs(theta) > 0.00001).sum(), theta.size)

    print ">> SGD"
    theta = theta0.copy()
    eta = 0.5
    t = time.time()
    for i in range(20):
        for fv in fvs:
            theta += eta * crf.gradient_likelihood([fv], theta)
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.95
    print "time = %.3f, relevant features = %d / %d" % (time.time() - t, (numpy.abs(theta) > 0.00001).sum(), theta.size)

    print ">> SGD + FOBOS L1"
    theta = theta0.copy()
    eta = 0.5
    lmd = 0.01
    t = time.time()
    for i in range(20):
        lmd_eta = lmd * eta
        for fv in fvs:
            theta += eta * crf.gradient_likelihood([fv], theta)
            theta = (theta > lmd_eta) * (theta - lmd_eta) + (theta < -lmd_eta) * (theta + lmd_eta)
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.95
    print "time = %.3f, relevant features = %d / %d" % (time.time() - t, (numpy.abs(theta) > 0.00001).sum(), theta.size)

    print ">> Steepest Descent + FOBOS L1"
    theta = theta0.copy()
    eta = 0.2
    lmd = 0.5
    t = time.time()
    for i in range(20):
        theta += eta * crf.gradient_likelihood(fvs, theta)
        lmd_eta = lmd * eta
        theta = (theta > lmd_eta) * (theta - lmd_eta) + (theta < -lmd_eta) * (theta + lmd_eta)
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.9
    print "time = %.3f, relevant features = %d / %d" % (time.time() - t, (numpy.abs(theta) > 0.00001).sum(), theta.size)
    #print theta

    print ">> BFGS"
    t = time.time()
    theta = crf.inference(fvs, theta0)
    print "log likelihood:", crf.likelihood(fvs, theta)
    print "time = %.3f, relevant features = %d / %d" % (time.time() - t, (numpy.abs(theta) > 0.00001).sum(), theta.size)


if __name__ == "__main__":
    main()

