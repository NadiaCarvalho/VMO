#!/usr/bin/env python3.7
"""
This script defines the Factor Oracle class
based on the code in https://github.com/wangsix/vmo/blob/master/vmo/VMO/oracle.py
"""

import numpy as np
from .utils import get_rsfx
from .factor_oracle import FactorOracle

class FO(FactorOracle):
    """ An implementation of the factor oracle
    """

    def __init__(self, **kwargs):
        super(FO, self).__init__(**kwargs)
        self.kind = 'r'

    def add_state(self, new_symbol, method=None):
        """
        :type self: oracle
        """
        self.basic_attributes['sfx'].append(0)
        self.basic_attributes['rsfx'].append([])
        self.basic_attributes['trn'].append([])
        self.basic_attributes['lrs'].append(0)
        self.basic_attributes['data'].append(new_symbol)

        self.statistics['n_states'] += 1

        i = self.statistics['n_states'] - 1

        self.basic_attributes['trn'][i - 1].append(i)
        k = self.basic_attributes['sfx'][i - 1]
        pi_1 = i - 1

        # Adding forward links
        while k is not None:
            _symbols = [self.basic_attributes['data'][state]
                        for state in self.basic_attributes['trn'][k]]
            if self.basic_attributes['data'][i] not in _symbols: # change here to use weights in checking if event is in
                self.basic_attributes['trn'][k].append(i)
                pi_1 = k
                k = self.basic_attributes['sfx'][k]
            else:
                break

        if k is None:
            self.basic_attributes['sfx'][i] = 0
            self.basic_attributes['lrs'][i] = 0
        else:
            _query = [[self.basic_attributes['data'][state], state] for state in
                      self.basic_attributes['trn'][k]
                      if self.basic_attributes['data'][state] == self.basic_attributes['data'][i]]
            _query = sorted(_query, key=lambda _query: _query[1])
            _state = _query[0][1]
            self.basic_attributes['sfx'][i] = _state
            self.basic_attributes['lrs'][i] = self._len_common_suffix(
                pi_1, self.basic_attributes['sfx'][i] - 1) + 1

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

    def accept(self, context):
        """ Check if the context could be accepted by the oracle

        Args:
            context: s sequence same type as the oracle data

        Returns:
            bAccepted: whether the sequence is accepted or not
            _next: the state where the sequence is accepted
        """
        _next = 0
        for _s in context:
            _data = [self.basic_attributes['data'][j]
                     for j in self.basic_attributes['trn'][_next]]
            if _s in _data:
                _next = self.basic_attributes['trn'][_next][_data.index(_s)]
            else:
                return 0, _next
        return 1, _next

    def get_alphabet(self):
        """get alphabet"""
        alphabet = [self.basic_attributes['data'][i]
                    for i in self.basic_attributes['trn'][0]]
        dictionary = dict(zip(alphabet, range(len(alphabet))))
        return dictionary

    @property
    def latent(self):
        """latency"""
        latent = []
        for symbol in self.basic_attributes['trn'][0]:
            indices = set([symbol])
            indices = get_rsfx(self, indices, symbol)
            latent.append(list(indices))
        return latent
