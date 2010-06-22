#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Conditional Random Field

import sys, os, re, pickle
from optparse import OptionParser

import numpy
from numpy.random import dirichlet, randn
from scipy import optimize


def load_text(filename, limit=1e9):
    labels = []
    text = []
    current_label = "H"
    f = open(filename, 'r')
    for line in f:
        m = re.match(r'##([A-Z])$', line)
        if m:
            current_label = m.group(1)
            continue
        text.append(line)
        labels.append(current_label)
        if len(text) >= limit: break
    f.close()
    return (text, labels)

class CRF(object):
    def __init__(self):
        self.features = []
        self.features_edge = []
        self.labels = ["start","stop","H","CR","B","I"]

    def add_feature(self, f):
        self.features.append(f)
    def add_feature_edge(self, f):
        self.features_edge.append(f)

    def init_param(self):
        self.theta = (numpy.random.rand(len(self.features)) - 0.5) / 10
        self.theta_edge = (numpy.random.rand(len(self.features_edge)) - 0.5) / 10

    def feature(self, x, y):
        energy = 0
        for x_i, y_i in zip(x, y):
            fv = [f(x_i, y_i) for f in self.features]
            energy += numpy.dot(self.theta, fv)
        print energy

    def M(self, i, x):
        '''calculate matrix M_i(x). (i=1,...,n-1)'''
        K = len(self.labels) # number of labels(states)
        M = numpy.zeros((K, K))
        for k2, y2 in enumerate(self.labels):

            fv = [f(x[i], y2) for f in self.features]
            fv = numpy.dot(self.theta, fv)

            for k1, y1 in enumerate(self.labels):
                gv = [g(x[i], x[i-1], y2, y1) for g in self.features_edge]
                M[k1, k2] = numpy.exp(numpy.dot(self.theta_edge, gv) + fv)
        return M


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="text filename")
    #parser.add_option("-k", dest="K", type="int", help="number of latent states", default=6)
    #parser.add_option("-a", dest="a", type="float", help="Dirichlet parameter", default=1.0)
    #parser.add_option("-i", dest="I", type="int", help="iteration count", default=10)
    #parser.add_option("-t", dest="triangle", action="store_true", help="triangle")
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    text, labels = load_text(options.filename, 20)

    crf = CRF()
    for label in crf.labels:
        for keyword in "project gutenberg etext copyright chapter".split():
            crf.add_feature( lambda x, y: 1 if re.search(keyword, x, re.I) and y == label else 0 )
        crf.add_feature_edge( lambda x, x_, y, y_: 1 if y == label else 0 )
        crf.add_feature_edge( lambda x, x_, y, y_: 1 if y_ == label else 0 )

    crf.add_feature( lambda x, y: 1 if re.search(r"[A-Z]{3}", x) else 0 )
    crf.add_feature( lambda x, y: 1 if re.search(r"[0-9]", x) else 0 )
    crf.add_feature( lambda x, y: 1 if x.find("@")>=0 else 0 )
    crf.add_feature( lambda x, y: 1 if x.find("#")>=0 else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("*") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("  ") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("    ") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("        ") else 0 )
    crf.add_feature( lambda x, y: 1 if len(x.strip()) == 0 else 0 )

    crf.add_feature_edge( lambda x, x_, y, y_: 1 if len(x_.strip()) == 0 else 0 )
    crf.add_feature_edge( lambda x, x_, y, y_: 1 if y == y_ else 0 )

    crf.init_param()
    #crf.feature(text[0:20], labels[0:20])

    alpha = [1 if y == "start" else 0 for y in crf.labels]
    Ms = []
    for i in range(1, len(text)):
        M = crf.M(i, text)
        Ms.append(M)
        alpha = numpy.dot(alpha, M)
    print alpha

    beta = [1 if y == "stop" else 0 for y in crf.labels]
    for i in range(len(Ms)):
        beta = numpy.dot(Ms[-i-1], beta)
    print beta
if __name__ == "__main__":
    main()

