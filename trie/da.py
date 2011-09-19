#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Double Array for static ordered data
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

class DoubleArray(object):
    def initialize(self, N=10000):
        self.base = [n - 1 for n in xrange(N)]
        self.check = [-n - 1 for n in xrange(N)]
        self.value = [0 for n in xrange(N)]
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

