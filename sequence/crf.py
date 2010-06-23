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

class CRF(object):
    def __init__(self):
        self.features = []
        self.features_edge = []
        self.labels = ["start","stop","H","B","F"] #,"CR","I"]

    def add_feature(self, f):
        self.features.append(f)
    def add_feature_edge(self, f):
        self.features_edge.append(f)

    def init_param(self):
        #self.theta = (numpy.random.rand(len(self.features)) - 0.5)
        #self.theta_edge = (numpy.random.rand(len(self.features_edge)) - 0.5)
        self.theta = numpy.random.randn(len(self.features))
        self.theta_edge = numpy.random.randn(len(self.features_edge))
        return numpy.append(self.theta, self.theta_edge)

    def set_param(self, theta):
        self.theta = theta[:len(self.theta)]
        self.theta_edge = theta[len(self.theta):]

    def feature(self, x, y):
        energy = 0
        for x_i, y_i in zip(x, y):
            fv = [f(x_i, y_i) for f in self.features]
            energy += numpy.dot(self.theta, fv)
        print energy

    def logM(self, i, x):
        '''calculate matrix M_i(x). (i=1,...,n-1)'''
        K = len(self.labels) # number of labels(states)
        logM = numpy.zeros((K, K))
        for k2, y2 in enumerate(self.labels):

            fv = [f(x[i], y2) for f in self.features]
            fv = numpy.dot(self.theta, fv)

            for k1, y1 in enumerate(self.labels):
                gv = [g(y1, y2) for g in self.features_edge]
                logM[k1, k2] = numpy.dot(self.theta_edge, gv) + fv
        return logM

    def cost(self, y, x):
        F = numpy.zeros(len(self.features), dtype=int)
        for y1, x1 in zip(y, x):
            F += [f(x1, y1) for f in self.features]
        G = numpy.zeros(len(self.features_edge), dtype=int)
        for y1, y2 in zip(["start"] + y, y + ["end"]):
            G += [g(y1, y2) for g in self.features_edge]
        return numpy.dot(self.theta, F) + numpy.dot(self.theta_edge, G)

    def likelihood(self, labels, text, theta=None):
        if theta!=None: self.set_param(theta)

        Ms = [self.logM(0, text)]
        logalpha = Ms[0][self.labels.index("start")] # alpha(1)
        #alpha = [1 if x=="start" else 0 for x in self.labels]
        #alpha = numpy.dot(alpha, numpy.exp(Ms[0]))
        #print numpy.exp(logalpha) / alpha - 1
        for i in range(1, len(text)): # n-2
            logM = self.logM(i, text)
            Ms.append(logM)
            logalpha = logdotexp_vec_mat(logalpha, logM)
            #alpha = numpy.dot(alpha, numpy.exp(logM))
            #print numpy.exp(logalpha) / alpha - 1

        #for m in Ms: print m

        logbeta = Ms[-1][:, self.labels.index("stop")]
        #beta = [1 if x=="stop" else 0 for x in self.labels]
        #beta = numpy.dot(numpy.exp(Ms[-1]), beta)
        #print numpy.exp(logbeta) / beta - 1

        for logM in Ms[-2::-1]:
            logbeta = logdotexp_mat_vec(logM, logbeta)
            #beta = numpy.dot(numpy.exp(logM), beta)
            #print numpy.exp(logbeta) / beta - 1

        #print self.labels
        #print "Z(X) = ", logalpha[self.labels.index("stop")], logbeta[self.labels.index("start")]
        #print logalpha / logbeta
        return self.cost(labels, text) - logalpha[self.labels.index("stop")]

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
    #print dict(map(lambda i: (i,1),labels)).keys()

    crf = CRF()
    for label in crf.labels:
        for keyword in "project gutenberg etext copyright chapter".split():
            crf.add_feature( lambda x, y: 1 if re.search(keyword, x, re.I) and y == label else 0 )
        crf.add_feature( lambda x, y: 1 if y == label else 0 )
        crf.add_feature_edge( lambda y_, y: 1 if y_ == label else 0 )

    crf.add_feature( lambda x, y: 1 if re.search(r"[A-Z]{3}", x) else 0 )
    crf.add_feature( lambda x, y: 1 if re.search(r"[0-9]", x) else 0 )
    crf.add_feature( lambda x, y: 1 if x.find("@")>=0 else 0 )
    crf.add_feature( lambda x, y: 1 if x.find("#")>=0 else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("*") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("  ") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("    ") else 0 )
    crf.add_feature( lambda x, y: 1 if x.startswith("        ") else 0 )
    crf.add_feature( lambda x, y: 1 if len(x.strip()) == 0 else 0 )

    #crf.add_feature_edge( lambda y_, y: 1 if len(x_.strip()) == 0 else 0 )
    crf.add_feature_edge( lambda y_, y: 1 if y == y_ else 0 )

    theta = crf.init_param()
    #crf.feature(text[0:20], labels[0:20])

    likelihood = lambda x:-crf.likelihood(labels, text, x)
    print -likelihood(theta)
    theta = optimize.fmin_bfgs(likelihood, theta)
    print -likelihood(theta)





if __name__ == "__main__":
    main()

