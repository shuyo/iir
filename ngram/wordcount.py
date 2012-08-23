#!/usr/bin/env python


import re, sys

class NaiveCounting:
    def __init__(self):
        self.map = dict()
    def add(self, word):
        if word in self.map:
            self.map[word] += 1
        else:
            self.map[word] = 1

class SpaceSaving:
    def __init__(self, k):
        self.k = k
        self.map = dict()
    def add(self, word):
        if word in self.map:
            self.map[word] += 1
        elif len(self.map) < self.k:
            self.map[word] = 1
        else:
            j = min(self.map, key=lambda x:self.map[x])
            cj = self.map.pop(j)
            self.map[word] = cj + 1


text = ""
for filename in sys.argv:
    with open(filename, "rb") as f:
        text += f.read()

c1 = NaiveCounting()
c2 = SpaceSaving(1000)
c3 = SpaceSaving(100)

n = 0
for m in re.finditer(r'[A-Za-z]+', text):
    word = m.group(0).lower()
    c1.add(word)
    c2.add(word)
    c3.add(word)
    n += 1

print "total words = %d" % n

words = c1.map.items()
words.sort(key=lambda x:(-x[1], x[0]))
m2 = c2.map
m3 = c3.map
for i, x in enumerate(words):
    print "%d\t%s\t%d\t%d\t%d" % (i+1, x[0], x[1], m2.get(x[0],0), m3.get(x[0],0))


