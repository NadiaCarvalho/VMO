#!/usr/bin/env python3.7
"""
This script defines the VMO (Variable Memory Oracle) class,
based on the code in https://github.com/wangsix/vmo/blob/master/vmo/VMO/oracle.py.
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
    VMO (Variable Memory Oracle) is a class that extends FactorOracle.
    It implements an oracle that creates transitions between states based on 
    distances between symbols and adds latent information for variable-length 
    patterns. It is mainly used for sequence modeling.

    Attributes:
        f_array (FeatureArray): The feature array that stores symbols in the oracle.
        latent (list): A list containing latent state information for compressed states.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VMO object with parameters passed via **kwargs.
        Resets the latent state information and initializes the feature array.
        
        Args:
            **kwargs: Arbitrary keyword arguments for configuring the VMO instance.
        """
        super(VMO, self).__init__(**kwargs)
        self.kind = 'a'

        self.f_array = FeatureArray(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))

        self.basic_attributes['data'][0] = None
        self.latent = []

    def reset(self, **kwargs):
        """
        Resets the VMO oracle by re-initializing its attributes and feature array.

        Args:
            **kwargs: Arbitrary keyword arguments for resetting the VMO instance.
        """
        super(VMO, self).reset(**kwargs)

        self.kind = 'a'

        self.f_array = FeatureArray(self.params['dim'])
        self.f_array.add(np.zeros(self.params['dim'], ))

        self.basic_attributes['data'][0] = None
        self.latent = []

    def _dvec(self, new_symbol, k):
        """
        Calculates the distance vector between a new symbol and previous states.

        Args:
            new_symbol (np.ndarray): The new symbol being added to the oracle.
            k (int): The index of the state to compute the distance from.

        Returns:
            np.ndarray: The computed distance vector.
        """
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
        """
        Handles state transitions and links when the complete method is used.
        
        The *complete* method ensures that the suffix links and transitions between states in the oracle are thoroughly created. 
        It does so by evaluating all possible suffix candidates and selecting the best one based on the computed distances between symbols.
        This makes it more computationally exhaustive than methods like "incremental," but it provides a more accurate model of the 
        state transitions and repeated patterns in the sequence data.

        Args:
            i (int): The current state index.
            pi_1 (int): The previous state index.
            suffix_candidate (list): A list of suffix candidates based on distance calculations. 
                - A suffix candidate is a possible link to an earlier state in the oracle that shares a suffix with the new state being added.
        """
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
        """
        Handles state transitions when the non-complete method is used.

        Args:
            k (int): The current suffix state.
            i (int): The current state index.
            pi_1 (int): The previous state index.
            suffix_candidate (int): The suffix candidate index.
        """
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
        """
        Makes temporary adjustments to the latent states and suffixes during backtracking.

        Args:
            i (int): The current state index.
        """
        k = self._find_better(
            i, self.basic_attributes['data'][i - self.basic_attributes['lrs'][i]])
        if k is not None:
            self.basic_attributes['lrs'][i] += 1
            self.basic_attributes['sfx'][i] = k

        self.basic_attributes['rsfx'][self.basic_attributes['sfx'][i]].append(i)

        if self.basic_attributes['lrs'][i] > self.statistics['max_lrs'][i - 1]:
            self.statistics['max_lrs'].append(self.basic_attributes['lrs'][i])
        else:
            self.statistics['max_lrs'].append(self.statistics['max_lrs'][i - 1])

        comp_1 = self.statistics['avg_lrs'][i - 1] * ((i - 1.0) / (self.statistics['n_states'] - 1.0))
        comp_2 = self.basic_attributes['lrs'][i] * (1.0 / (self.statistics['n_states'] - 1.0))
        self.statistics['avg_lrs'].append(comp_1 + comp_2)

    def add_state(self, new_symbol, method='inc', verbose=False):
        """
        Adds a new state to the VMO, updating state transitions, suffixes, and latent state information.

        Args:
            new_symbol (np.ndarray): The new symbol to be added.
            method (str): The method used for state update ('inc' for incremental or 'complete').
            verbose (bool): Whether to print timing information for state addition.
        """
        start_time = time.time()
        self.basic_attributes['sfx'].append(0)
        self.basic_attributes['rsfx'].append([])
        self.basic_attributes['trn'].append([])
        self.basic_attributes['lrs'].append(0)

        self.f_array.add(new_symbol)

        self.statistics['n_states'] += 1
        i = self.statistics['n_states'] - 1

        self.basic_attributes['trn'][i - 1].append(i)
        k = self.basic_attributes['sfx'][i - 1]
        pi_1 = i - 1

        suffix_candidate = (0, [])[method == 'complete']

        while k is not None:
            dvec = self._dvec(new_symbol, k)
            suffix = np.where(dvec < self.params['threshold'])[0]

            if len(suffix) == 0:
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
                new_s = self.basic_attributes['trn'][k][suffix[np.argmin(dvec[suffix])]]
                suffix_candidate.append((new_s, np.min(dvec)))
            else:
                suffix_candidate = self.basic_attributes['trn'][k][suffix[np.argmin(dvec[suffix])]]
                break

            if method == 'complete':
                k = self.basic_attributes['sfx'][k]

        if method == 'complete':
            self._complete_method(i, pi_1, suffix_candidate)
        else:
            self._non_complete_method(k, i, pi_1, suffix_candidate)

        self._temporary_adjustment(i)

        if verbose:
            print(f"I {self.statistics['n_states']} --- {time.time() - start_time} seconds ---")
            
