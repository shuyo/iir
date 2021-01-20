#!/usr/bin/env python
# -*- coding: utf-8 -

# Randomized Response (Gibbs Sampling)
# This code is available under the MIT License.
# (c)2021 Nakatani Shuyo / Cybozu Labs Inc.

nlist = [100, 1000, 10000]

import numpy
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=2, suppress=True)

true_prob = numpy.array([0.1, 0.2, 0.3, 0.4])
#true_prob /= true_prob.sum()
D = true_prob.size
legend = [str(x) for x in true_prob]

p = 1/5
pii = (1 + (D - 1) * p) / D # P(Y=i|X=i)
pij = (1 - p) / D # P(Y=j|X=i)
P = pij * numpy.ones((D, D)) + (pii - pij) * numpy.eye(D)

def gibbs_sampling(N, alpha):
    true_count = numpy.array(N * true_prob, dtype=int)
    true_count[-1] += N - true_count.sum()

    predicts = []
    for _ in range(10000):
        c = sum(numpy.random.multinomial(n, P[i,:]) for i, n in enumerate(true_count))

        pi = numpy.ones(D) + numpy.random.random(D) # initial
        pi /= pi.sum()
        sample = []
        for epoch in range(400):
            Q = pi * P.T # _ij = pi_j * P_ji
            cond = Q.T / Q.sum(axis=1) # _ij = pi_i * P_ij / Σ_k pi_k * P_kj

            # sampling X
            X = numpy.sum([numpy.random.multinomial(n, cond[:,i]) for i, n in enumerate(c)], axis=0)

            # sampling pi
            pi = numpy.random.dirichlet(alpha + X)

            if epoch >= 200: sample.append(X)

        predicts.append(numpy.mean(sample, axis=0)/N)

    return numpy.array(predicts)

for N in nlist:
    for alpha in [1.0, 0.1, 0.01]:
        predicts = gibbs_sampling(N, alpha)
        start = predicts.min()
        end = predicts.max()
        bins = 40
        step = (end - start)/bins

        plt.hist(predicts, bins=numpy.arange(start, end, step), density=True)
        plt.title("N = %d, alpha = %.2f" % (N, alpha))
        plt.legend(legend)
        plt.tight_layout()
        plt.savefig("rr-gibbs-%d-%.2f.png" % (N, alpha))
        plt.close()

        print("N=%d, alpha=%.2f, 1.true, 2.mean, 3.std, 4-5.95%%, 6.median" % (N, alpha))
        print(numpy.vstack((
            [true_prob, numpy.mean(predicts, axis=0), numpy.std(predicts, axis=0)],
            numpy.quantile(predicts, [0.025,0.975,0.5], axis=0))))
