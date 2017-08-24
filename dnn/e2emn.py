#!/usr/bin/env python

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

re_q = re.compile(r'\?\s(.+)\s(\d+)$')
class CorpusLoader(object):
    def __init__(self):
        self.vocab = Vocab()
        self.vocab_a = Vocab()
    def load(self, *files):
        lines = []
        toid = lambda x: numpy.array([self.vocab[y] for y in x.split()])
        for path in files:
            knowledge = []
            with open(path) as f:
                for s in f:
                    s = re.sub(r'^\d+ ', '', s.strip().lower())
                    s = re.sub(r'([\.\?,])', r' \1', s)
                    m = re_q.search(s)
                    s = re_q.sub('?', s)

                    if m:
                        a = m.group(1)
                        #strict_supervised = [int(x) for x in m.group(2).split()]
                        lines.append((knowledge, toid(s), self.vocab_a[a.strip()])) # (x_nij, q_nj, a_n)
                        knowledge = []
                    else:
                        knowledge.append(toid(s))
        return Corpus(lines)

class Corpus(object):
    def __init__(self, lines):
        self._batchsize = 25
        self.lines = lines

    @property
    def batchsize(self): return self._batchsize
    @batchsize.setter
    def batchsize(self, bs): self._batchsize = bs

    def __iter__(self):
        idx = numpy.random.permutation(len(self))
        i = 0
        while i<len(self):
            js = idx[i:i+self._batchsize]
            yield [self.lines[j][0] for j in js], [self.lines[j][1] for j in js], numpy.array([self.lines[j][2] for j in js])
            i += self._batchsize

    def __len__(self):
        return len(self.lines)

# End-to-End Memory Network Model
# simple implementation (suppose that each knowledge has the same number of sentences in a minibatch)
class E2EMN(chainer.Chain):
    def __init__(self, vocab_s, vocab_a, D):
        super(E2EMN, self).__init__()
        with self.init_scope():
            self.embedid_a = L.EmbedID(vocab_s, D)
            self.embedid_b = L.EmbedID(vocab_s, D)
            self.embedid_c = L.EmbedID(vocab_s, D)
            self.W = L.Linear(D, vocab_a)

    def forward(self, x, q): # (x_nij, q_nj) -> ahat (unnormalized log prob)
        M = F.stack([F.stack([F.sum(self.embedid_a(xi), axis=0) for xi in xn]) for xn in x])
        U = F.stack([F.sum(self.embedid_b(qj), axis=0) for qj in q])
        C = F.stack([F.stack([F.sum(self.embedid_c(xi), axis=0) for xi in xn]) for xn in x])

        P = F.softmax(F.batch_matmul(M, U)[:, :, 0])
        O = F.batch_matmul(F.swapaxes(C, 1, 2), P)[:, :, 0]

        return self.W(O+U)

    def __call__(self, x, q, a):
        ahat = self.forward(x, q)
        return F.softmax_cross_entropy(ahat, a), ahat

corpus = CorpusLoader()
dir = "tasks_1-20_v1-2/en/"
target = "qa1_single-supporting-fact"
#target = "qa9_simple-negation"
train_data = corpus.load(dir+target+"_train.txt")
test_data = corpus.load(dir+target+"_test.txt")
print(len(train_data), len(corpus.vocab), len(corpus.vocab_a))

model = E2EMN(len(corpus.vocab), len(corpus.vocab_a), 150)
optimizer = chainer.optimizers.Adam()
#optimizer = chainer.optimizers.SGD(0.001)
optimizer.setup(model)

t0 = time.time()
for epoch in range(50):
    train_loss = 0
    train_correct = 0
    for x, q, a in train_data:
        model.cleargrads()
        loss, ahat = model(x, q, a)
        train_loss += loss.data
        train_correct += (ahat.data.argmax(axis=1)==a).sum()
        loss.backward()
        optimizer.update()

    test_loss = 0
    test_correct = 0
    with chainer.no_backprop_mode():
        for x, q, a in test_data:
            loss, ahat = model(x, q, a)
            test_loss += loss.data
            test_correct += (ahat.data.argmax(axis=1)==a).sum()

    print(epoch, time.time()-t0, train_loss / len(train_data), train_correct / len(train_data), test_loss / len(test_data), test_correct / len(test_data))

