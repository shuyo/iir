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
        self.N, self.K = tlist.shape
        self.PHI = numpy.apply_along_axis(feature_func, 1, a)
        self.M = self.PHI.shape[1]
        self.w = numpy.random.randn(self.M, self.K)

    def predict(self, phi):
        pass

    def inference(self):
        """learning once iteration"""
        pass

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    (options, args) = parser.parse_args()

    if not options.filename: parser.error("need corpus filename(-f)")
    corpus = vocabulary.load_file(options.filename)
    if options.seed != None:
        numpy.random.seed(options.seed)


if __name__ == "__main__":
    main()
