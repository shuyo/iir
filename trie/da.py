#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy

# Double Array for static ordered data
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

class DoubleArray(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def validate_list(self, list):
        pre = ""
        for i, line in enumerate(list):
            if pre >= line:
                raise Exception, "list has not ascent order at %d" % (i+1)
            pre = line

    def initialize(self, list):
        self.validate_list(list)

        self.N = 1
        self.base  = [-1]
        self.check = [-1]
        self.value = [-1]

        max_index = 0
        queue = collections.deque([(0, 0, len(list), 0)])
        while len(queue) > 0:
            index, left, right, depth = queue.popleft()
            if depth >= len(list[left]):
                self.value[index] = left
                left += 1
                if left >= right: continue

            # get branches of current node
            stack = collections.deque([(right, -1)])
            cur, c1 = (left, ord(list[left][depth]))
            result = []
            while len(stack) >= 1:
                while c1 == stack[-1][1]:
                    cur, c1 = stack.pop()
                mid = (cur + stack[-1][0]) / 2
                if cur == mid:
                    result.append((cur + 1, c1))
                    cur, c1 = stack.pop()
                else:
                    c2 = ord(list[mid][depth])
                    if c1 != c2:
                        stack.append((mid, c2))
                    else:
                        cur = mid

            # search empty index for current node
            v0 = result[0][1]
            j = - self.check[0] - v0
            while any(j + v < self.N and self.check[j + v] >= 0 for right, v in result):
                j = - self.check[j + v0] - v0
            tail_index = j + result[-1][1]
            if max_index < tail_index:
                max_index = tail_index
                self.extend_array(tail_index + 2)

            # insert current node into DA
            self.base[index] = j
            depth += 1
            for right, v in result:
                child = j + v
                self.check[self.base[child]] = self.check[child]
                self.base[-self.check[child]] = self.base[child]
                self.check[child] = index
                queue.append((child, left, right, depth))
                left = right

        self.shrink_array(max_index)

    def extend_array(self, max_cand):
        if self.N < max_cand:
            new_N = 2 ** int(numpy.ceil(numpy.log2(max_cand)))
            self.log("extend DA : %d => (%d) => %d", (self.N, max_cand, new_N))
            self.base.extend(    n - 1 for n in xrange(self.N, new_N))
            self.check.extend( - n - 1 for n in xrange(self.N, new_N))
            self.value.extend(     - 1 for n in xrange(self.N, new_N))
            self.N = new_N

    def shrink_array(self, max_index):
        self.log("shrink DA : %d => %d", (self.N, max_index + 1))
        self.N = max_index + 1
        self.check = numpy.array(self.check[:self.N])
        self.base = numpy.array(self.base[:self.N])
        self.value = numpy.array(self.value[:self.N])

        not_used = self.check < 0
        self.check[not_used] = -1
        not_used[0] = False
        self.base[not_used] = self.N

    def log(self, format, param):
        if self.verbose:
            import time
            print "-- %s, %s" % (time.strftime("%Y/%m/%d %H:%M:%S"), format % param)

    def save(self, filename):
        numpy.savez(filename, base=self.base, check=self.check, value=self.value)

    def load(self, filename):
        loaded = numpy.load(filename)
        self.base = loaded['base']
        self.check = loaded['check']
        self.value = loaded['value']
        self.N = self.base.size

    def add_element(self, s, v):
        pass

    def get_subtree(self, s):
        cur = 0
        for c in iter(s):
            v = ord(c)
            next = self.base[cur] + v
            if next >= self.N or self.check[next] != cur:
                return None
            cur = next
        return cur

    def get_child(self, c, subtree):
        v = ord(c)
        next = self.base[subtree] + v
        if next >= self.N or self.check[next] != subtree:
            return None
        return next

    def get(self, s):
        cur = self.get_subtree(s)
        if cur >= 0:
            value = self.value[cur]
            if value >= 0: return value
        return None

    def get_value(self, subtree):
        return self.value[subtree]

    def extract_features(self, st):
        events = dict()
        pointers = []
        for c in iter(st):
            pointers.append(0)
            new_pointers = []
            for pointer in pointers:
                p = self.get_child(c, pointer)
                if p is not None:
                    new_pointers.append(p)
                    id = self.value[p]
                    if id >= 0:
                        events[id] = events.get(id, 0) + 1
            pointers = new_pointers
        return events



