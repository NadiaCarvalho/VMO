#!/usr/bin/env python3.7
"""
This script defines the FeatureArray class
based on the code in https://github.com/wangsix/vmo/blob/master/vmo/VMO/oracle.py
"""

import numpy as np

class FeatureArray:
    def __init__(self, dim):
        self.data = np.zeros((100, dim))
        self.dim = dim
        self.capacity = 100
        self.size = 0

    def __getitem__(self, item):
        return self.data[item, :]

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity, self.dim))
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = x
        self.size += 1

    def finalize(self):
        self.data = self.data[:self.size, :]
