#!/usr/bin/env python3.7
"""
This script defines the VMO oracle class
based on the code in https://github.com/wangsix/vmo/blob/master/vmo/VMO/oracle.py
"""

import math
import time

import numpy as np
import scipy.spatial.distance as dist

from .factor_oracle import FactorOracle
from ..feature_array import FeatureArray
from ..cdist_fixed import fixed_cdist

class VMO(FactorOracle):
    """
    Class VMO
    """

    def __init__(self, **kwargs):
        super(VMO, self).__init__(**kwargs)
        self.kind = 'a'

        self.f_array = FeatureArray(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))

        self.basic_attributes['data'][0] = None
        self.latent = []

    def reset(self, **kwargs):
        super(VMO, self).reset(**kwargs)

        self.kind = 'a'

        self.f_array = FeatureArray(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))

        self.basic_attributes['data'][0] = None
        self.latent = []

    def _dvec(self, new_symbol, k):
        """docstring"""
        if self.params['dfunc'] == 'other':
            return dist.cdist([new_symbol],
                              self.f_array[self.basic_attributes['trn'][k]],
                              metric=self.params['dfunc_handle'], w=self.params['weights'])[0]
        if self.params['weights'] is not None and self.params['fixed_weights'] is not None:
            return fixed_cdist([new_symbol],
                               self.f_array[self.basic_attributes['trn'][k]],
                               metric=self.params['dfunc'],
                               w=self.params['weights'],
                               fw=self.params['fixed_weights'])[0]
        if self.params['weights'] is not None:
            return dist.cdist([new_symbol],
                              self.f_array[self.basic_attributes['trn'][k]],
                              metric=self.params['dfunc'], w=self.params['weights'])[0]
        return dist.cdist([new_symbol],
                          self.f_array[self.basic_attributes['trn'][k]],
                          metric=self.params['dfunc'])[0]

    def _complete_method(self, i, pi_1, suffix_candidate):
        """docstring"""
        if not suffix_candidate:
            self.basic_attributes['sfx'][i] = 0
            self.basic_attributes['lrs'][i] = 0
            self.latent.append([i])
            self.basic_attributes['data'].append(len(self.latent) - 1)
        else:
            sorted_suffix_candidates = sorted(
                suffix_candidate, key=lambda suffix: suffix[1])
            self.basic_attributes['sfx'][i] = sorted_suffix_candidates[0][0]
            self.basic_attributes['lrs'][i] = self._len_common_suffix(
                pi_1, self.basic_attributes['sfx'][i] - 1) + 1
            self.latent[self.basic_attributes['data']
                        [self.basic_attributes['sfx'][i]]].append(i)
            self.basic_attributes['data'].append(
                self.basic_attributes['data'][self.basic_attributes['sfx'][i]])

    def _non_complete_method(self, k, i, pi_1, suffix_candidate):
        """docstring"""
        if k is None:
            self.basic_attributes['sfx'][i] = 0
            self.basic_attributes['lrs'][i] = 0
            self.latent.append([i])
            self.basic_attributes['data'].append(len(self.latent) - 1)
        else:
            self.basic_attributes['sfx'][i] = suffix_candidate
            self.basic_attributes['lrs'][i] = self._len_common_suffix(
                pi_1, self.basic_attributes['sfx'][i] - 1) + 1
            self.latent[self.basic_attributes['data']
                        [self.basic_attributes['sfx'][i]]].append(i)
            self.basic_attributes['data'].append(
                self.basic_attributes['data'][self.basic_attributes['sfx'][i]])

    def _temporary_adjustment(self, i):
        k = self._find_better(
            i, self.basic_attributes['data'][i - self.basic_attributes['lrs'][i]])
        if k is not None:
            self.basic_attributes['lrs'][i] += 1
            self.basic_attributes['sfx'][i] = k

        self.basic_attributes['rsfx'][self.basic_attributes['sfx'][i]].append(
            i)

        if self.basic_attributes['lrs'][i] > self.statistics['max_lrs'][i - 1]:
            self.statistics['max_lrs'].append(self.basic_attributes['lrs'][i])
        else:
            self.statistics['max_lrs'].append(
                self.statistics['max_lrs'][i - 1])

        comp_1 = self.statistics['avg_lrs'][i - 1] * \
            ((i - 1.0) / (self.statistics['n_states'] - 1.0))
        comp_2 = self.basic_attributes['lrs'][i] * \
            (1.0 / (self.statistics['n_states'] - 1.0))
        self.statistics['avg_lrs'].append(comp_1 + comp_2)

    def add_state(self, new_symbol, method='inc', verbose=False):
        start_time = time.time()
        """Create new state and update related links and compressed state"""
        self.basic_attributes['sfx'].append(0)
        self.basic_attributes['rsfx'].append([])
        self.basic_attributes['trn'].append([])
        self.basic_attributes['lrs'].append(0)

        # Experiment with pointer-based
        self.f_array.add(new_symbol)

        self.statistics['n_states'] += 1
        i = self.statistics['n_states'] - 1

        # assign new transition from state i-1 to i
        self.basic_attributes['trn'][i - 1].append(i)
        k = self.basic_attributes['sfx'][i - 1]
        pi_1 = i - 1

        # iteratively backtrack suffixes from state i-1
        suffix_candidate = (0, [])[method == 'complete']

        while k is not None:
            dvec = self._dvec(new_symbol, k)
            suffix = np.where(dvec < self.params['threshold'])[0]

            if len(suffix) == 0:  # if no transition from suffix
                # Add new forward link to unvisited state
                self.basic_attributes['trn'][k].append(i)
                pi_1 = k
                if method != 'complete':
                    k = self.basic_attributes['sfx'][k]
            elif method == 'inc':
                new_s = suffix[0]
                if suffix.shape[0] != 1:
                    new_s = suffix[np.argmin(dvec[suffix])]
                suffix_candidate = self.basic_attributes['trn'][k][new_s]
                break
            elif method == 'complete':
                new_s = self.basic_attributes['trn'][k][suffix[np.argmin(
                    dvec[suffix])]]
                suffix_candidate.append((new_s, np.min(dvec)))
            else:
                suffix_candidate = self.basic_attributes['trn'][k][suffix[np.argmin(
                    dvec[suffix])]]
                break

            if method == 'complete':
                k = self.basic_attributes['sfx'][k]

        if method == 'complete':
            self._complete_method(i, pi_1, suffix_candidate)
        else:
            self._non_complete_method(k, i, pi_1, suffix_candidate)

        # Temporary adjustment
        self._temporary_adjustment(i)

        if verbose:
            print("I %i --- %s seconds ---" % (self.statistics['n_states'], time.time() - start_time))

        #image = gen_plot.start_draw(self)
        #name = r'data\oracles\oracle_at_' + str(i) + '.PNG'
        #image.save(name)

