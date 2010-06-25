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
        return [fv + numpy.dot(theta_g, gm) for gm in self.Gmat] + [fv]

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
    def __init__(self, features, regularity, sigma=1):
        self.features = features
        if regularity == 0:
            self.regularity = lambda w:0
            self.regularity_deriv = lambda w:0
        elif regularity == 1:
            self.regularity = lambda w:numpy.sum(numpy.abs(w)) / sigma
            self.regularity_deriv = lambda w:numpy.sign(w) / sigma
        else:
            v = sigma ** 2
            v2 = v * 2
            self.regularity = lambda w:numpy.sum(w ** 2) / v2
            self.regularity_deriv = lambda w:numpy.sum(w) / v
        #self.pre_theta = numpy.zeros(self.features.size()+ self.features.size_edge())

    def random_param(self):
        return numpy.random.randn(self.features.size()+ self.features.size_edge())

    def logalphas(self, Mlist):
        logalpha = Mlist[0][self.features.start_label_index()] # alpha(1)
        logalphas = [logalpha]
        for logM in Mlist[1:]:
            logalpha = logdotexp_vec_mat(logalpha, logM)
            logalphas.append(logalpha)
        return logalphas

    def logbetas(self, Mlist):
        logbeta = Mlist[-1][:, self.features.stop_label_index()]
        logbetas = [logbeta]
        for logM in Mlist[-2::-1]:
            logbeta = logdotexp_mat_vec(logM, logbeta)
            logbetas.append(logbeta)
        return logbetas[::-1]

    def likelihood(self, fv, theta):
        '''conditional log likelihood log p(Y|X)'''
        #print "T" if (theta == self.pre_theta).all() else "F"
        #self.pre_theta = theta

        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])

        log_Z = self.logalphas(logMlist)[-1][self.features.stop_label_index()]
        return fv.cost(theta)  - self.regularity(theta) - log_Z

    def gradient_likelihood(self, fv, theta):
        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])
        logalphas = self.logalphas(logMlist)
        logbetas = self.logbetas(logMlist)
        log_Z = logalphas[-1][self.features.stop_label_index()]

        grad = numpy.array(fv.Fss, dtype=float) # empirical expectation

        expect_edge = numpy.zeros_like(logMlist[0])
        for i in range(len(logMlist)):
            if i==0:
                expect_edge[self.features.start_label_index()] += numpy.exp(logalphas[i] + logbetas[i+1] - log_Z)
            elif i<len(logbetas)-1:
                m = logMlist[i]
                a = logalphas[i-1][:,numpy.newaxis]
                b = logbetas[i+1]
                expect_edge += numpy.exp(m + b + a - log_Z)
            else:
                expect_edge[:,self.features.stop_label_index()] += numpy.exp(logalphas[i-1] + logbetas[i] - log_Z)
        for k, indexes in enumerate(fv.Fon[0]):
            #print grad[k], expect_edge, indexes
            grad[k] -= numpy.sum(expect_edge.take(indexes))

        for i, gm in enumerate(fv.Gmat):
            p_yi = numpy.exp(logalphas[i] + logbetas[i+1] - log_Z)
            #print i, p_yi, sum(p_yi), gm
            grad[n_fe:] -= numpy.sum(gm * numpy.exp(logalphas[i] + logbetas[i+1] - log_Z), axis=1)

        return grad - self.regularity_deriv(theta)


def logdotexp_vec_mat(loga, logM):
    return numpy.array([maxentropy.logsumexp(loga + x) for x in logM.T], copy=False)
def logdotexp_mat_vec(logM, logb):
    return numpy.array([maxentropy.logsumexp(x + logb) for x in logM], copy=False)

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

    fv = FeatureVector(features, labels, text)

    crf = CRF(features, options.regularity)
    likelihood = lambda x:-crf.likelihood(fv, x)
    likelihood_deriv = lambda x:-crf.gradient_likelihood(fv, x)

    theta = crf.random_param()
    print "theta:", theta
    print "-log likelihood:", likelihood(theta)

    theta = optimize.fmin_bfgs(likelihood, theta, fprime=likelihood_deriv)
    print likelihood(theta), theta


if __name__ == "__main__":
    main()

