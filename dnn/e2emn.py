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

toarray = numpy.array

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

    def load(self, *files, device=-1):
        xp = chainer.cuda.cupy if device>=0 else numpy
        toa = lambda a: xp.array(a, dtype=numpy.int32)
        toid = lambda x: [self.vocab[y] for y in x.split()]

        lines = []
        knowledge = []
        for path in files:
            with open(path) as f:
                startid = 0
                for s in f:
                    if s.startswith("1 "): startid = len(knowledge)
                    s = re.sub(r'^\d+ ', '', s.strip().lower())
                    s = re.sub(r'([\.\?,])', r'', s)
                    m = s.split("\t")

                    if len(m)==3:
                        #strict_supervised = [int(x) for x in m[2].split()]
                        lines.append(((startid, len(knowledge)), toid(m[0]), self.vocab_a[m[1]])) # (x_nij, q_nj, a_n)
                        #if len(lines)>=100: break
                    else:
                        knowledge.append(toid(s))
        D = len(self.vocab)
        X = xp.zeros((len(knowledge), D*2), dtype=numpy.float32)
        for i, x in enumerate(knowledge):
            J = len(x)
            for j, z in enumerate(x):
                X[i, z] += 1
                X[i, z+D] += j/J
        Q = xp.zeros((len(lines), D), dtype=numpy.float32)
        for i, x in enumerate(lines):
            for z in x[1]: Q[i, z] += 1
        return Corpus(X, Q, numpy.array([x[0] for x in lines]), xp.array([x[2] for x in lines], dtype=numpy.int32))

class Corpus(object):
    def __init__(self, x, q, k, a):
        self.knowledge = x
        self.query = q
        self.kindex = k
        self.answer = a

    def __iter__(self):
        K, _ = self.kindex.shape
        idx = numpy.random.permutation(K)
        for i in idx:
            k = self.kindex[i]
            yield self.knowledge[k[0]:k[1]], self.query[i], self.answer[i:i+1]

    def __len__(self):
        return self.answer.size

    @property
    def ksize(self):
        return self.knowledge.shape[0]

class E2EMN(chainer.Chain):
    def __init__(self, layer, D, vocab, vocab_ans, max_knowledge, pe=False, rn=False):
        super(E2EMN, self).__init__()
        self.layer = layer
        self.V = vocab
        self.pe = pe # Position Encoding
        self.rn = rn # Random Noise

        initializer = chainer.initializers.Normal(0.1)
        with self.init_scope():
            self.embedid_a = chainer.Parameter(initializer, (vocab, D))
            self.embedid_b = chainer.Parameter(initializer, (vocab, D))
            self.embedid_c = chainer.Parameter(initializer, (vocab, D))
            self.W = L.Linear(D, vocab_ans, initialW=initializer)
            self.temporal_a = chainer.Parameter(initializer, (max_knowledge, D))
            self.temporal_c = chainer.Parameter(initializer, (max_knowledge, D))
            if layer > 1:
                self.H = L.Linear(D, D, initialW=initializer)

    # (x_ij, q_j) -> a^hat (unnormalized log probabilites)
    def forward(self, x, q, is_linear=False):
        # Random Noise for Learing Time invariance
        if chainer.configuration.config.train:
            xp = chainer.cuda.get_array_module(x)
            z = xp.zeros((1,x.shape[1]), dtype=numpy.float32)
            i = 0
            while i<x.shape[0]:
                if numpy.random.rand(1)[0]<0.1:
                    x = xp.vstack((x[:i], z, x[i:]))
                    i += 1
                i += 1
        max_knowledge, D = self.temporal_a.shape
        if len(x)>max_knowledge: x = x[len(x)-max_knowledge:]
        j = max_knowledge-len(x)

        if self.pe:
            a = xp.arange(1,0,-1/D)
            b = xp.arange(-1,1,2/D)
            M = a * F.matmul(x[:,:self.V], self.embedid_a) + b * F.matmul(x[:,self.V:], self.embedid_a) + self.temporal_a[j:]
            C = a * F.matmul(x[:,:self.V], self.embedid_c) + b * F.matmul(x[:,self.V:], self.embedid_c) + self.temporal_c[j:]
        else:
            M = F.matmul(x[:,:self.V], self.embedid_a) + self.temporal_a[j:]
            C = F.matmul(x[:,:self.V], self.embedid_c) + self.temporal_c[j:]

        U = F.matmul(q.reshape(1,-1), self.embedid_b)
        for l in range(self.layer):
            P = F.transpose(F.matmul(M,U[0]))
            if not is_linear: P = F.softmax(P)
            O = F.matmul(P,C)
            if l == self.layer-1:
                U = U + O
            else:
                U = self.H(U) + O
        return self.W(U) # (1,D)

    def __call__(self, x, q, a, is_linear=False):
        ahat = self.forward(x, q, is_linear)
        return F.softmax_cross_entropy(ahat, a), ahat

def main():
    import argparse
    parser = argparse.ArgumentParser(description='End-to-End Memory Network')
    parser.add_argument('-l', '--layer', help='number of layers', type=int, default=1)
    parser.add_argument('-d', '--dim', help='dimension of hidden unit', type=int, default=100)
    parser.add_argument('-e', '--epoch', help='epoches', type=int, default=100)
    parser.add_argument('-t', '--target', help='target data', default="tasks_1-20_v1-2/en/qa1_single-supporting-fact")
    parser.add_argument('--adam', help='use Adam optimizer', action="store_true")
    parser.add_argument('--pe', help='use Position Encoding', action="store_true")
    parser.add_argument('--ls', help='use Linear Start', action="store_true")
    parser.add_argument('--rn', help='use Random Noise', action="store_true")
    parser.add_argument("-g", "--gpu", default=-1, type=int, help="GPU ID (negative = CPU)")
    args = parser.parse_args()
    print(args)

    corpus = CorpusLoader()
    train_data = corpus.load(args.target+"_train.txt", device=args.gpu)
    valid_data = corpus.load(args.target+"_test.txt", device=args.gpu)
    print("knowledge=%d, query=%d, vocab=%d, answer=%d" %
        (train_data.ksize, len(train_data), len(corpus.vocab), len(corpus.vocab_a)))

    model = E2EMN(args.layer, args.dim, len(corpus.vocab), len(corpus.vocab_a), 100, pe=args.pe, rn=args.rn)
    xp = numpy
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy

    if args.adam:
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.SGD(0.01)
    optimizer.setup(model)

    t0 = time.time()
    linear_start = args.ls
    min_valid_loss = 1e9
    loss_raise_count = 0
    for epoch in range(args.epoch):
        train_loss = 0
        train_correct = 0
        for x, q, a in train_data:
            model.cleargrads()
            loss, ahat = model(x, q, a, linear_start)
            train_loss += loss.data
            if ahat.data.argmax()==a: train_correct += 1
            loss.backward()
            for p in model.params():
                n2 = xp.linalg.norm(p.grad)
                if n2>40:
                    #print("[warning : L2 norm of grad > 40 (%s)]" % p.name)
                    p.grad *= 40/n2
            optimizer.update()

        valid_loss = 0
        valid_correct = 0
        with chainer.no_backprop_mode():
            for x, q, a in valid_data:
                loss, ahat = model(x, q, a, linear_start)
                valid_loss += loss.data
                if ahat.data.argmax()==a: valid_correct += 1

        print("%d\t%.1f\t%.3f\t%f\t%.3f\t%f" % (epoch, time.time()-t0, train_loss / len(train_data), train_correct / len(train_data), valid_loss / len(valid_data), valid_correct / len(valid_data)))

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            loss_raise_count = 0
        elif linear_start:
            loss_raise_count += 1
            if loss_raise_count>=3:
                print("[Linear Start : re-insert softmax layer]")
                linear_start = False

if __name__=="__main__":
    main()

