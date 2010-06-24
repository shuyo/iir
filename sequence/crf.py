#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Conditional Random Field

import sys, os, re, pickle
from optparse import OptionParser

import numpy
from scipy import optimize, maxentropy
numpy.set_printoptions(precision=5)


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

class FeatureVector(object):
    def __init__(self, features, ylist, xlist):
        '''statistics of features (sufficient statistics like)'''
        flist = features.features
        glist = features.features_edge
        self.K = len(features.labels)

        self.Fss = numpy.zeros(len(flist), dtype=int)
        for y1, x1 in zip(ylist, xlist):
            self.Fss += [f(x1, y1) for f in flist]

        self.Gss = numpy.zeros(len(glist), dtype=int)
        for y1, y2 in zip(["start"] + ylist, ylist + ["end"]):
            self.Gss += [g(y1, y2) for g in glist]

        self.Fmat = []
        self.Gmat = []
        for x in xlist:
            mt = numpy.zeros((len(flist), self.K), dtype=int)
            for j, f in enumerate(flist):
                mt[j] = [f(x, y) for y in features.labels]
            self.Fmat.append(mt)

            gmlist = []
            for y1 in features.labels:
                mt = numpy.zeros((len(glist), self.K), dtype=int)
                for j, g in enumerate(glist):
                    mt[j] = [g(y1, y2) for y2 in features.labels]
                gmlist.append(mt)
            self.Gmat.append(gmlist)

    def cost(self, theta_f, theta_g):
        return numpy.dot(theta_f, self.Fss) + numpy.dot(theta_g, self.Gss)

    def logMlist(self, theta_f, theta_g):
        Mlist = []
        for fm, gmlist in zip(self.Fmat, self.Gmat):
            gv = numpy.zeros((self.K, self.K))
            for j, gm in enumerate(gmlist):
                gv[j] = numpy.dot(theta_g, gm)
            Mlist.append(gv + numpy.dot(theta_f, fm))
        return Mlist


class Features(object):
    def __init__(self, labels):
        self.features = []
        self.features_edge = []
        self.labels = ["start","stop"] + dict(map(lambda i: (i,1),labels)).keys()
    def start_label_index(self):
        return 0
    def stop_label_index(self):
        return 1
    def size(self):
        return len(self.features)
    def size_edge(self):
        return len(self.features_edge)

    def add_feature(self, f):
        self.features.append(f)
    def add_feature_edge(self, f):
        self.features_edge.append(f)

class CRF(object):
    def __init__(self, features):
        self.features = features

    def random_param(self):
        return numpy.random.randn(self.features.size()+ self.features.size_edge())

    def likelihood(self, fv, theta):
        theta_f = theta[:self.features.size()]
        theta_g = theta[self.features.size():]

        Mlist = fv.logMlist(theta_f, theta_g)
        logalpha = Mlist[0][self.features.start_label_index()] # alpha(1)
        for logM in Mlist[1:]: # n-2
            logalpha = logdotexp_vec_mat(logalpha, logM)

        #logbeta = Mlist[-1][:, self.features.stop_label_index()]
        #for logM in Mlist[-2::-1]:
        #    logbeta = logdotexp_mat_vec(logM, logbeta)

        return fv.cost(theta_f, theta_g) - logalpha[self.features.stop_label_index()]

def logdotexp_vec_mat(loga, logM):
    return numpy.array([maxentropy.logsumexp(loga + x) for x in logM.T], copy=False)
def logdotexp_mat_vec(logM, logb):
    return numpy.array([maxentropy.logsumexp(x + logb) for x in logM], copy=False)

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="text filename")
    #parser.add_option("-k", dest="K", type="int", help="number of latent states", default=6)
    #parser.add_option("-a", dest="a", type="float", help="Dirichlet parameter", default=1.0)
    #parser.add_option("-i", dest="I", type="int", help="iteration count", default=10)
    #parser.add_option("-t", dest="triangle", action="store_true", help="triangle")
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
    features.add_feature_edge( lambda y_, y: 1 if y == y_ else 0 )

    fv = FeatureVector(features, labels, text)

    crf = CRF(features)
    likelihood = lambda x:-crf.likelihood(fv, x)

    minL = 1e9
    for i in range(10):
        theta1 = crf.random_param()
        L = likelihood(theta1)
        print L
        if minL > L:
            minL = L
            theta = theta1
    print likelihood(theta), theta
    theta = optimize.fmin_bfgs(likelihood, theta)
    print likelihood(theta), theta


if __name__ == "__main__":
    main()

