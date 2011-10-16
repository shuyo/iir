#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

# Double Array for static ordered data
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

class DoubleArray(object):
    def validate_list(self, list):
        pre = ""
        for line in list:
            if pre >= line:
                raise ""

    def initialize(self, list):
        self.validate_list(list)

        self.N = 0
        self.base = []
        self.check = []
        self.value = []

        max_index = 0
        queue = collections.deque([(0, 0, len(list), 0)])
        while len(queue) > 0:
            index, left, right, depth = queue.popleft()
            if depth >= len(list[left]):
                value[index] = left
                left += 1
                if left >= right: continue

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

            v0 = result[0][1]
            self.extend_array(max_index + result[-1][1] - v0 + 3)

            j = - self.check[0] - v0
            while any(self.check[j+v] >= 0 for right, v in result):
                j = - self.check[j+v0] - v0
            self.base[index] = j

            depth += 1
            for right, v in result:
                child = j + v
                self.check[self.base[child]] = self.check[child]
                self.base[-self.check[child]] = self.base[child]
                self.check[child] = index
                queue.append((child, left, right, depth))
                left = right
            if child > max_index: max_index = child
        self.shrink_array(max_index)

    def extend_array(self, max_cand):
        if self.N < max_cand:
            new_N = 2 ** int(numpy.ceil(numpy.log2(max_cand + 1)))
            self.log("extend DA : %d => (%d) => %d" % (self.N, max_cand, new_N))
            self.base.extend(    n - 1 for n in xrange(self.N, new_N))
            self.check.extend( - n - 1 for n in xrange(self.N, new_N))
            self.value.extend(     - 1 for n in xrange(self.N, new_N))
            self.N = new_N

    def shrink_array(self, max_index):
        pass

    def log(self, st):
        print "-- %s, %s" % (time.strftime("%Y/%m/%d %H:%M:%S"), st)



    def add_element(self, s, v):
        x = self.root
        for c in s:
            if c not in x: x[c] = dict()
            x = x[c]
        x[""] = v


    def get_subtree(self, s):
        x = self.root
        for c in iter(st):
            if c not in x: return None
            x = x[c]
        return x
    def get_child(self, c, subtree):
        if c not in x: return None
        return subtree[c]
    def get(self, s):
        return self.get_value(self.get_subtree(s))
    def get_value(self, subtree):
        return subtree[""]

