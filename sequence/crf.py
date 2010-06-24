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
        flist = features.features_edge
        glist = features.features
        self.K = len(features.labels)

        # expectation of features under empirical distribution
        self.Fss = numpy.zeros(len(flist) + len(glist), dtype=int)
        for y1, y2 in zip(["start"] + ylist, ylist + ["end"]):
            self.Fss[:len(flist)] += [f(y1, y2) for f in flist]
        for y1, x1 in zip(ylist, xlist):
            self.Fss[len(flist):] += [g(x1, y1) for g in glist]

        # index list of ON values of edge features
        self.Fon = [] # (n, #f, indexes)

        # for calculation of M_i
        self.Fmat = [] # (n, K, #f, K)-matrix
        self.Gmat = [] # (n, #g, K)-matrix
        for x in xlist:
            mt = numpy.zeros((len(glist), self.K), dtype=int)
            for j, g in enumerate(glist):
                mt[j] = [g(x, y) for y in features.labels]  # sparse
            self.Gmat.append(mt)

            # when fmlist depends on x_i (if necessary)
            #self._calc_fmlist(flist, x)

        # when fmlist doesn't depend on x_i
        self._calc_fmlist(features)


    def _calc_fmlist(self, features):
        flist = features.features_edge
        fmlist = []
        f_on = [[] for f in flist]
        for k1, y1 in enumerate(features.labels):
            mt = numpy.zeros((len(flist), self.K), dtype=int)
            for j, f in enumerate(flist):
                mt[j] = [f(y1, y2) for y2 in features.labels]  # sparse
                f_on[j].extend([k1 * self.K + k2 for k2, v in enumerate(mt[j]) if v == 1])
            fmlist.append(mt)
        self.Fmat.append(fmlist)
        self.Fon.append(f_on)

    def cost(self, theta):
        return numpy.dot(theta, self.Fss)

    def logMlist(self, theta_f, theta_g):
        '''for independent fmlists on x_i'''
        fv = numpy.zeros((self.K, self.K))
        for j, fm in enumerate(self.Fmat[0]):
            fv[j] = numpy.dot(theta_f, fm)
        return [fv + numpy.dot(theta_g, gm) for gm in self.Gmat]

    def logMlist2(self, theta_f, theta_g):
        '''for dependent fmlists on x_i'''
        Mlist = []
        for gm, fmlist in zip(self.Gmat, self.Fmat):
            fv = numpy.zeros((self.K, self.K))
            for j, fm in enumerate(fmlist):
                fv[j] = numpy.dot(theta_f, fm)
            Mlist.append(fv + numpy.dot(theta_g, gm))
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
        #self.pre_theta = numpy.zeros(self.features.size()+ self.features.size_edge())

    def random_param(self):
        return numpy.random.randn(self.features.size()+ self.features.size_edge())

    def logalpha(self, Mlist):
        logalpha = Mlist[0][self.features.start_label_index()] # alpha(1)
        for logM in Mlist[1:]: # n-2
            logalpha = logdotexp_vec_mat(logalpha, logM)
        return logalpha

    def logbeta(self, Mlist):
        logbeta = Mlist[-1][:, self.features.stop_label_index()]
        for logM in Mlist[-2::-1]:
            logbeta = logdotexp_mat_vec(logM, logbeta)
        return logbeta

    def likelihood(self, fv, theta):
        '''conditional log likelihood log p(Y|X)'''
        #print "T" if (theta == self.pre_theta).all() else "F"
        #self.pre_theta = theta

        n_fe = self.features.size_edge() # number of features on edge
        Mlist = fv.logMlist(theta[:n_fe], theta[n_fe:])

        log_Z = self.logalpha(Mlist)[self.features.stop_label_index()]
        return fv.cost(theta) - log_Z

    def gradient_likelihood(self, fv, theta):
        grad = self.Fss # empirical expectation

        n_fe = self.features.size_edge() # number of features on edge
        Mlist = fv.logMlist(theta[:n_fe], theta[n_fe:])
        logalpha = self.logalpha(Mlist)
        logbeta = self.logbeta(Mlist)

        for i, indexes in enumerate(fv.Fon[0]):
            pass
            #grad[:n_fe] -=

        for i, gm in enumerate(fv.Gmat):
            pass
            #grad[n_fe:] -=

        # L2-regurality
        pass

        return grad


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

