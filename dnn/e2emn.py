#!/usr/bin/env python

import re
import numpy
import chainer
import chainer.functions as F
import chainer.links as L

re_q = re.compile(r'\?\s(.+)\s(\d+)$')
class Corpus(object):
    def __init__(self, *files):
        self._batchsize = 25
        self.lines = []
        self.vocab = []
        self.ids = dict()
        toid = lambda x: numpy.array([self[y] for y in x.split()])
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
                        #a_num = int(m.group(2))
                        self.lines.append((knowledge, toid(s), self[a.strip()])) # (x_nij, q_nj, a_n)
                        knowledge = []
                    else:
                        knowledge.append(toid(s))

    def __getitem__(self, w):
        if w not in self.ids:
            self.ids[w] = len(self.vocab)
            self.vocab.append(w)
        return self.ids[w]

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

# End-to-End Memory Network
class E2EMN(chainer.Chain):
    def __init__(self, vocab, D):
        super(E2EMN, self).__init__()
        with self.init_scope():
            self.embedid_a = L.EmbedID(vocab, D)
            self.embedid_b = L.EmbedID(vocab, D)
            self.embedid_c = L.EmbedID(vocab, D)
            self.W = L.Linear(D, vocab)

    def forward(self, x, q): # (x_nij, q_nj) -> ahat (unnormalized log prob)
        M = F.stack([F.stack([F.sum(self.embedid_a(xi), axis=0) for xi in xn]) for xn in x])
        U = F.stack([F.sum(self.embedid_b(qj), axis=0) for qj in q])
        C = F.stack([F.stack([F.sum(self.embedid_c(xi), axis=0) for xi in xn]) for xn in x])

        P = F.softmax(F.batch_matmul(M, U)[:, :, 0])
        O = F.batch_matmul(F.swapaxes(C, 1, 2), P)[:, :, 0]

        return self.W(O+U)

    def __call__(self, x, q, a):
        ahat = self.forward(x, q)
        return F.softmax_cross_entropy(ahat, a)

train_data = Corpus("tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt")
test_data = Corpus("tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt")
#print(len(train_data))

model = E2EMN(len(train_data.vocab), 100)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

for epoch in range(10):
    total_loss = 0
    for x, q, a in train_data:
        model.cleargrads()
        loss = model(x, q, a)
        total_loss += loss.data
        loss.backward()
        optimizer.update()

    print(epoch, total_loss / len(train_data))

