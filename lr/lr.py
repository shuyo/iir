#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Logistic Regression + Steepest Descent + FOBOS L1
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy

def linear_features():
    return lambda x: numpy.concatenate(([1], x))

class LR:
    def __init__(self, xlist, tlist, feature_func):
        self.tlist = tlist
        self.N, self.K = tlist.shape
        self.PHI = numpy.apply_along_axis(feature_func, 1, xlist)
        self.M = self.PHI.shape[1]
        self.w = numpy.random.randn(self.K, self.M)

    def predict(self, phi):
        y = numpy.dot(self.w, phi)
        y = numpy.exp(y - y.max())
        return y / y.sum()

    def error(self, phi, t):
        return -numpy.log(numpy.inner(self.predict(phi), t))

    def error_full(self):
        return numpy.sum([self.error(phi, self.tlist[n,]) for n, phi in enumerate(self.PHI)])

    def error_gradient(self, phi, t):
        return numpy.outer(self.predict(phi) - t, phi)

    def error_gradient_full(self):
        return numpy.sum([self.error_gradient(phi, self.tlist[n,]) for n, phi in enumerate(self.PHI)], axis=0)

    def inference_SGD(self, eta, reg=0, reg_coef=0.1):
        """SGD iteration once"""
        for n in numpy.random.shuffle(numpy.arange(self.N)):
            phi = self.PHI[n,]
            self.w -= eta * self.error_gradient(phi, self.tlist[n,])
            if reg == 1:
                self.fobos_l1(eta * reg_coef)
            elif reg == 2:
                self.fobos_l2(eta * reg_coef)

    def inference_SD(self, eta, reg=0, reg_coef=0.1):
        """Steepest Descent iteration once"""
        self.w -= eta * self.error_gradient_full()
        if reg == 1:
            self.fobos_l1(eta * reg_coef)
        elif reg == 2:
            self.fobos_l2(eta * reg_coef)

    def fobos_l1(self, reg_coef):
        self.w = (self.w - reg_coef) * (self.w > reg_coef) + (self.w + reg_coef) * (self.w < -reg_coef)

    def fobos_l2(self, reg_coef):
        self.w /= 1.0 + reg_coef

def main():
    import optparse
    parser = optparse.OptionParser()
    #parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("-r", dest="reg", type="int", help="regularization(r=0,1,2)", default=0)
    parser.add_option("-l", dest="reg_coef", type="float", help="regularization coefficient", default=0.1)
    parser.add_option("--sgd", dest="sgd", help="SGD inference", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (options, args) = parser.parse_args()

    #if not options.filename: parser.error("need corpus filename(-f)")
    #corpus = vocabulary.load_file(options.filename)
    if options.seed != None:
        numpy.random.seed(options.seed)

    iris_data = numpy.loadtxt('../data/iris.data.txt', delimiter=",", usecols=(0,1,2,3))
    iris_label = numpy.loadtxt('../data/iris.data.txt', delimiter=",", usecols=(4,), dtype='S16')

    # normalization
    xlist = iris_data - numpy.mean(iris_data, axis=0)
    xlist /= numpy.std(xlist, axis=0)

    # target as 1 of K representation
    labels = {'Iris-setosa':[1,0,0], 'Iris-versicolor':[0,1,0], 'Iris-virginica':[0,0,1]}
    tlist = numpy.array([labels[x] for x in iris_label])

    lr = LR(xlist, tlist, linear_features())

    eta = 0.1
    reg_coef = options.reg_coef
    for i in range(options.iteration):
        if options.sgd:
            lr.inference_SGD(eta, options.reg, reg_coef)
        else:
            lr.inference_SD(eta, options.reg, reg_coef)

        print lr.error_full()
    print lr.w

if __name__ == "__main__":
    main()
