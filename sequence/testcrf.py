#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project Gutenberg Content Extractor with CRF

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
    print "log likelihood:", crf.likelihood(fvs, theta0)

    print ">> BFGS"
    t = time.time()
    theta = theta0.copy()
    theta2 = crf.inference(fvs, theta)
    print "log likelihood:", crf.likelihood(fvs, theta2)
    print "time:", (time.time() - t)

    print ">> Steepest Descent"
    t = time.time()
    theta = theta0.copy()
    for i in range(10):
        grad = crf.gradient_likelihood(fvs, theta)
        #print "gradient:", grad
        theta += 0.2 * grad
        print i, "log likelihood:", crf.likelihood(fvs, theta)
    print "time:", (time.time() - t)

    print ">> SGD"
    t = time.time()
    theta = theta0.copy()
    eta = 0.5
    for i in range(10):
        for fs in fvs:
            grad = crf.gradient_likelihood([fv], theta)
            theta += eta * grad
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.95
    print "time:", (time.time() - t)

    print ">> SGD + FOBOS L1"
    t = time.time()
    theta = theta0.copy()
    eta = 0.5
    lmd = 0.01
    for i in range(10):
        lmd_eta = lmd * eta
        for fs in fvs:
            grad = crf.gradient_likelihood([fv], theta)
            theta += eta * grad
            for i, w in enumerate(theta):
                if w > 0:
                    if w > lmd_eta:
                        theta[i] = w - lmd_eta
                    else:
                        theta[i] = 0
                elif w < 0:
                    if w < -lmd_eta:
                        theta[i] = w + lmd_eta
                    else:
                        theta[i] = 0
        print i, "log likelihood:", crf.likelihood(fvs, theta)
        eta *= 0.95
    print "time:", (time.time() - t)

    print ">> Steepest Descent + FOBOS L1"
    t = time.time()
    theta = theta0.copy()
    eta = 0.2
    lmd = 0.1
    for i in range(20):
        lmd_eta = lmd * eta
        grad = crf.gradient_likelihood(fvs, theta)
        #print "gradient:", grad
        theta += eta * grad
        for i, w in enumerate(theta):
            if w > 0:
                if w > lmd_eta:
                    theta[i] = w - lmd_eta
                else:
                    theta[i] = 0
            elif w < 0:
                if w < -lmd_eta:
                    theta[i] = w + lmd_eta
                else:
                    theta[i] = 0
        print i, "log likelihood:", crf.likelihood(fvs, theta)
    print "time:", (time.time() - t)
    print theta


def hoge():
    print "features:", features.size()
    print "labels:", len(features.labels)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(fvs, theta)
    prob, ys = crf.tagging(text_fv, theta)
    print "tagging:", prob, features.id2label(ys)

    theta = crf.inference(fvs, theta)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(fvs, theta)
    prob, ys = crf.tagging(text_fv, theta)
    print "tagging:", prob, zip(test_texts[0], test_labels[0], features.id2label(ys))
    print theta

if __name__ == "__main__":
    main()

