#!/usr/bin/env python
# -*- coding: utf-8 -

# Randomized Response (Maximum Likelihood Estimation)
# This code is available under the MIT License.
# (c)2021 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=2,suppress=True)

true_prob = numpy.array([0.1, 0.2, 0.3, 0.4])
#true_prob /= true_prob.sum()
D = true_prob.size
legend = [str(x) for x in true_prob]

p = 1/5
pii = (1 + (D - 1) * p) / D # P(Y=i|X=i)
pij = (1 - p) / D # P(Y=j|X=i)
P = pij * numpy.ones((D, D)) + (pii - pij) * numpy.eye(D)

for N in [100, 1000, 10000]:
    true_count = numpy.array(N * true_prob, dtype=int)
    true_count[-1] += N - true_count.sum()

    predicts = []
    for _ in range(10000):
        c = sum(numpy.random.multinomial(n, P[i,:]) for i, n in enumerate(true_count)) # Randomized Response
        t = numpy.linalg.solve(P, c) # MLE
        predicts.append(t/N)
    predicts = numpy.array(predicts)

    start = predicts.min()
    end = predicts.max()
    bins = 40
    step = (end - start)/bins

    plt.hist(predicts, bins=numpy.arange(start, end, step), density=True)
    plt.title("N = %d" % N)
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig("rr-mle-%d.png" % N)
    plt.close()

    print("N=%d, 1.true, 2.mean, 3.std, 4-5.95%%, 6.median" % N)
    print(numpy.vstack((
        [true_prob, numpy.mean(predicts, axis=0), numpy.std(predicts, axis=0)],
        numpy.quantile(predicts, [0.025,0.975,0.5], axis=0))))

