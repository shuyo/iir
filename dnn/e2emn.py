#!/usr/bin/env python

# Simple Implementation of End-to-End Memory Network on Python3/Chainer2

# Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "End-to-end memory networks." Advances in neural information processing systems. 2015.

# This code is available under the MIT License.
# (c)2017 Nakatani Shuyo, Cybozu Labs Inc.

import re, time
import numpy
import chainer
import chainer.functions as F
import chainer.links as L

class Vocab(object):
    def __init__(self):
        self.vocab = []
        self.ids = dict()
    def __getitem__(self, w):
        if w not in self.ids:
            self.ids[w] = len(self.vocab)
            self.vocab.append(w)
        return self.ids[w]
    def __len__(self):
        return len(self.vocab)

class CorpusLoader(object):
    def __init__(self):
        self.vocab = Vocab()
        self.vocab_a = Vocab()

    def load(self, *files):
        lines = []
        toid = lambda x: numpy.array([self.vocab[y] for y in x.split()], dtype=numpy.int32)
        knowledge_size = 0
        for path in files:
            knowledge = []
            with open(path) as f:
                for s in f:
                    s = re.sub(r'^\d+ ', '', s.strip().lower())
                    s = re.sub(r'([\.\?,])', r'', s)
                    m = s.split("\t")

                    if len(m)==3:
                        #strict_supervised = [int(x) for x in m[2].split()]
                        lines.append((knowledge, toid(m[0]), self.vocab_a[m[1]])) # (x_nij, q_nj, a_n)
                        knowledge_size += len(knowledge)
                        #if len(lines)>=100: break
                        knowledge = []
                    else:
                        knowledge.append(toid(s))
        return Corpus(lines, knowledge_size)

class Corpus(object):
    def __init__(self, lines, knowledge_size):
        self.lines = lines
        self.knowledge_size = knowledge_size

    def __iter__(self):
        xx = []
        for x, q, a in self.lines:
            xx.extend(x)
            yield xx, q, a

    def __len__(self):
        return len(self.lines)

class E2EMN(chainer.Chain):
    def __init__(self, layer, D, vocab, vocab_ans, max_knowledge):
        super(E2EMN, self).__init__()
        self.layer = layer
        initializer = chainer.initializers.Normal(0.1)
        with self.init_scope():
            self.embedid_a = L.EmbedID(vocab, D, initialW=initializer)
            self.embedid_b = L.EmbedID(vocab, D, initialW=initializer)
            self.embedid_c = L.EmbedID(vocab, D, initialW=initializer)
            self.W = L.Linear(D, vocab_ans, initialW=initializer)
            self.temporal_a = chainer.Parameter(initializer, (max_knowledge, D))
            self.temporal_c = chainer.Parameter(initializer, (max_knowledge, D))
            if layer > 1:
                self.H = L.Linear(D, D, initialW=initializer)

    def forward(self, x, q): # (x_ij, q_j) -> a^hat (unnormalized log prob)
        max_knowledge, _ = self.temporal_a.shape
        if len(x)>max_knowledge: x = x[len(x)-max_knowledge:]
        M = F.stack([F.sum(self.embedid_a(xi), axis=0) for xi in x])
        C = F.stack([F.sum(self.embedid_c(xi), axis=0) for xi in x])

        j = max_knowledge-len(x)
        M += self.temporal_a[j:]
        C += self.temporal_c[j:]

        U = F.sum(self.embedid_b(q), axis=0).reshape(1,-1)
        for l in range(self.layer):
            P = F.matmul(M,U[0])
            O = F.matmul(F.transpose(P),C)
            if l == self.layer-1:
                U = U + O
            else:
                U = self.H(U) + O
        return self.W(U) # (1,D)

    def __call__(self, x, q, a):
        ahat = self.forward(x, q)
        return F.softmax_cross_entropy(ahat, numpy.array([a], dtype=numpy.int32)), ahat

def main():
    import argparse
    parser = argparse.ArgumentParser(description='End-to-End Memory Network')
    parser.add_argument('-l', '--layer', help='number of layers', type=int, default=1)
    parser.add_argument('-d', '--dim', help='dimension of hidden unit', type=int, default=100)
    parser.add_argument('-e', '--epoch', help='epoches', type=int, default=50)
    parser.add_argument('-t', '--target', help='target data', default="tasks_1-20_v1-2/en/qa1_single-supporting-fact")
    parser.add_argument('--adam', help='use Adam optimizer', action="store_true")
    args = parser.parse_args()
    print(args)

    corpus = CorpusLoader()
    train_data = corpus.load(args.target+"_train.txt")
    test_data = corpus.load(args.target+"_test.txt")
    print(len(train_data), len(corpus.vocab), len(corpus.vocab_a))

    model = E2EMN(args.layer, args.dim, len(corpus.vocab), len(corpus.vocab_a), 100)
    if args.adam:
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.SGD(0.001)
    optimizer.setup(model)

    t0 = time.time()
    for epoch in range(args.epoch):
        train_loss = 0
        train_correct = 0
        n = 0
        for x, q, a in train_data:
            model.cleargrads()
            loss, ahat = model(x, q, a)
            train_loss += loss.data
            if ahat.data.argmax()==a: train_correct += 1
            loss.backward()
            optimizer.update()
            n += 1
            if n % 200 == 0: print(n, train_loss/n, train_correct/n)

        test_loss = 0
        test_correct = 0
        with chainer.no_backprop_mode():
            for x, q, a in test_data:
                loss, ahat = model(x, q, a)
                test_loss += loss.data
                if ahat.data.argmax()==a: test_correct += 1

        print(epoch, time.time()-t0, train_loss / len(train_data), train_correct / len(train_data), test_loss / len(test_data), test_correct / len(test_data))

if __name__=="__main__":
    main()

