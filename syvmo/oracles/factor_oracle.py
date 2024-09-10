#!/usr/bin/env python3.7
"""
This script defines the Factor Oracle class,
as the base class for the FO(factor oracle) and VMO(variab_le markov oracle)
based on the code in https://github.com/wangsix/vmo/b_lob/master/vmo/VMO/oracle.py
"""

import numpy as np
from .utils import entropy


class FactorOracle:
    """ The base class for the FO(factor oracle) and VMO(variab_le markov oracle)

    Attributes:
        sfx: a list containing the suffix link of each state.
        trn: a list containing the forward links of each state as a list.
        rsfx: a list containing the reverse suffix links of each state
            as a list.
        lrs: the value of longest repeated suffix of each state.
        data: the symbols associated with the direct link
            connected to each state.
        compror: a list of tuples (i, i-j), i is the current coded position,
            i-j is the length of the corresponding coded words.
        code: a list of tuples (len, pos), len is the length of the
            corresponding coded words, pos is the position where the coded
            words starts.
        seg: same as code but non-overlapping.
        f_array: (For kind 'a' and 'v'): a list containing the feature array
        latent: (For kind 'a' and 'v'): a list of lists with each sub-list
            containing the indexes for each symbol.
        kind:
            'a': Variable Markov oracle
            'f': repeat oracle
            'v': Centroid-based oracle (under test)
        n_states: number of total states, also is length of the input
            sequence plus 1.
        max_lrs: the longest lrs so far.
        avg_lrs: the average lrs so far.
        name: the name of the oracle.
        params: a python dictionary for different feature and distance settings.
            keys:
                'thresholds': the minimum value for separating two values as
                    different symbols.
                'weights': a dictionary containing different weights for features
                    used.
                'dfunc': the distance function.
    """

    def __init__(self, **kwargs):
        # Basic attributes
        self.basic_attributes = {
            'sfx': [],
            'trn': [],
            'rsfx': [],
            'lrs': [],
            'data': []
        }

        # Compression attributes
        self.comp_attributes = {
            'compror': [],
            'code': [],
            'seg': []
        }

        # Object attributes
        self.obj_attributes = {
            'kind': 'f',
            'name': ''
        }

        # Oracle statistics
        self.statistics = {
            'n_states': 1,
            'max_lrs': [],
            'avg_lrs': []
        }
        self.statistics['max_lrs'].append(0)
        self.statistics['avg_lrs'].append(0.0)

        # Oracle parameters
        self.params = {
            'threshold': 0,
            'dfunc': 'cosine',
            'dfunc_handle': None,
            'dim': 1,
            'weights': None,
            'fixed_weights': None,
        }
        self.update_params(**kwargs)

        # Adding zero state
        self.basic_attributes['sfx'].append(None)
        self.basic_attributes['rsfx'].append([])
        self.basic_attributes['trn'].append([])
        self.basic_attributes['lrs'].append(0)
        self.basic_attributes['data'].append(0)

    def reset(self, **kwargs):
        """
        Reset parameters back to default values
        """
        self.update_params(**kwargs)
        # Basic attributes
        self.basic_attributes['sfx'] = []
        self.basic_attributes['trn'] = []
        self.basic_attributes['rsfx'] = []
        self.basic_attributes['lrs'] = []
        self.basic_attributes['data'] = []

        # Compression attributes
        self.comp_attributes['compror'] = []
        self.comp_attributes['code'] = []
        self.comp_attributes['seg'] = []

        # Object attributes
        self.obj_attributes['kind'] = 'f'
        self.obj_attributes['name'] = ''

        # Oracle statistics
        self.statistics['n_states'] = 1
        self.statistics['max_lrs'] = []
        self.statistics['max_lrs'].append(0)
        self.statistics['avg_lrs'] = []
        self.statistics['avg_lrs'].append(0.0)

        # Adding zero state
        self.basic_attributes['sfx'].append(None)
        self.basic_attributes['rsfx'].append([])
        self.basic_attributes['trn'].append([])
        self.basic_attributes['lrs'].append(0)
        self.basic_attributes['data'].append(0)

    def update_params(self, **kwargs):
        """
        Updates the oracle parameters with the provided values.
    
        Parameters:
            **kwargs: Key-value pairs to update the oracle's parameters.
        """
        self.params.update(kwargs)

    def add_state(self, new_symbol, method=None):
        """
        Adds a new state to the oracle by associating it with a new symbol.
    
        Parameters:
            new_symbol: The symbol to associate with the new state.
            method: Optional method to determine how to add the state.
        """
        self.basic_attributes['data'].append(new_symbol)

    def _condition(self, i, j):
        """
        Checks whether a given state satisfies whether a certain range or segment (from j to i) 
        is sufficiently represented in the next state based on the length of its longest repeated suffix. 
        
        The method _condition returns True when:
        - i is not the last state.
        - The next state i + 1 has a repeated suffix that is at least as long as the current segment from j to i.

        If both conditions are met, the method can proceed to encode or link states.

        Parameters:
            i: The index of the current state.
            j: The index of the previous state.
    
        Returns:
            bool: True if the condition is met, False otherwise.
        """

        return ((i < self.statistics['n_states'] - 1)
                and (self.basic_attributes['lrs'][i + 1] >= i - j + 1))

    def _encode(self):
        """
        Internal routine for encoding the structure of the oracle using compression.
    
        Returns:
            tuple: A tuple containing the code and compression ratio for the oracle.
        """

        _code = []
        _compror = []

        if not self.comp_attributes['compror']:
            j = 0
        else:
            j = self.comp_attributes['compror'][-1][0]

        i = j
        while j < self.statistics['n_states'] - 1:
            while self._condition(i, j):
                i += 1
            if i == j:
                i += 1
                _code.append([0, i])
                _compror.append([i, 0])
            else:
                _code.append(
                    [i - j, self.basic_attributes['sfx'][i] - i + j + 1])
                _compror.append([i, i - j])
            j = i
        return _code, _compror

    def encode(self):
        """
        Encodes the oracle's structure and updates the compression attributes.
    
        Returns:
            tuple: A tuple containing the oracle's code and compression ratio.
        """
        _c, _cmpr = self._encode()
        self.comp_attributes['code'].extend(_c)
        self.comp_attributes['compror'].extend(_cmpr)
        return self.comp_attributes['code'], self.comp_attributes['compror']

    def segment(self):
        """
        Generates a non-overlapping version of the compression ratio (Compror).
    
        Returns:
            list: A list of segments representing the oracle's non-overlapping compression.
        """
        if not self.comp_attributes['seg']:
            j = 0
        else:
            j = self.comp_attributes['seg'][-1][1]
            last_len = self.comp_attributes['seg'][-1][0]
            if last_len + j > self.statistics['n_states']:
                return None

        i = j
        while j < self.statistics['n_states'] - 1:
            while self._condition(i, j):
                i += 1
            if i == j:
                i += 1
                self.comp_attributes['seg'].append((0, i))
            else:
                if (self.basic_attributes['sfx'][i] + self.basic_attributes['lrs'][i]) <= i:
                    self.comp_attributes['seg'].append(
                        (i - j, self.basic_attributes['sfx'][i] - i + j + 1))

                else:
                    _i = j + i - self.basic_attributes['sfx'][i]
                    self.comp_attributes['seg'].append(
                        (_i - j, self.basic_attributes['sfx'][i] - i + j + 1))
                    _j = _i
                    num = self.basic_attributes['lrs'][_i +
                                                       1] - self.basic_attributes['lrs'][_j]
                    while (_i < i) and (num >= _i - _j + 1):
                        _i += 1
                    if _i == _j:
                        _i += 1
                        self.comp_attributes['seg'].append((0, _i))
                    else:
                        self.comp_attributes['seg'].append(
                            (_i - _j, self.basic_attributes['sfx'][_i] - _i + _j + 1))
            j = i
        return self.comp_attributes['seg']

    def _ir(self, alpha=1.0):
        """
        Computes the information rate (IR) based on the code words and entropy.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
    
        Returns:
            tuple: A tuple containing the information rate, h_0, and h_1 values.
        """
        code, _ = self.encode()
        c_w = np.zeros(len(code))  # Number of code words
        for i, char in enumerate(code):
            c_w[i] = char[0] + 1

        c_0 = [1 if x[0] == 0 else 0 for x in self.comp_attributes['code']]
        h_0 = np.log2(np.cumsum(c_0))

        h_1 = np.zeros(len(c_w))

        for i in range(1, len(c_w)):
            h_1[i] = entropy(c_w[0:i + 1])

        i_r = alpha * h_0 - h_1

        return i_r, h_0, h_1

    def _ir_fixed(self, alpha=1.0):
         """
        Computes a fixed information rate (IR) based on the number of clusters and maximum lrs.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
    
        Returns:
            tuple: A tuple containing the fixed information rate, h_0, and h_1 values.
        """
        code, _ = self.encode()

        h_0 = np.log2(self.num_clusters())

        if self.statistics['max_lrs'][-1] == 0:
            h_1 = np.log2(self.statistics['n_states'] - 1)
        else:
            h_1 = np.log2(self.statistics['n_states'] - 1) + \
                np.log2(self.statistics['max_lrs'][-1])

        b_l = np.zeros(self.statistics['n_states'] - 1)
        j = 0
        for i, _ in enumerate(code):
            if self.comp_attributes['code'][i][0] == 0:
                b_l[j] = 1
                j += 1
            else:
                loc = code[i][0]
                b_l[j:j + loc] = loc  # range(1,loc+1)
                j = j + loc
        i_r = alpha * h_0 - h_1 / b_l
        i_r[i_r < 0] = 0
        return i_r, h_0, h_1

    def _ir_cum(self, alpha=1.0):
        """
        Computes the cumulative information rate (IR) by considering the appearance of new states.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
    
        Returns:
            tuple: A tuple containing the cumulative information rate, h_0, and h_1 values.
        """
        code, _ = self.encode()
        states = self.statistics['n_states']

        # c_w0 counts the appearance of new states only
        c_w0 = np.zeros(states - 1)
        # c_w1 counts the appearance of all compror states
        c_w1 = np.zeros(states - 1)
        # b_l is the b_lock length of compror codewords
        b_l = np.zeros(states - 1)

        j = 0
        for i, _ in enumerate(code):
            if self.comp_attributes['code'][i][0] == 0:
                c_w0[j] = 1
                c_w1[j] = 1
                b_l[j] = 1
                j += 1
            else:
                loc = code[i][0]
                c_w1[j] = 1
                b_l[j:j + loc] = loc  # range(1,loc+1)
                j = j + loc

        h_0 = np.log2(np.cumsum(c_w0))
        h_1 = np.log2(np.cumsum(c_w1))
        h_1 = h_1 / b_l
        i_r = alpha * h_0 - h_1
        i_r[i_r < 0] = 0

        return i_r, h_0, h_1

    def _ir_cum2(self, alpha=1.0):
        """
        Computes a second version of cumulative information rate (IR) based on state counts.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
    
        Returns:
            tuple: A tuple containing the cumulative information rate, h_0, and h_1 values.
        """
        code, _ = self.encode()
        states = self.statistics['n_states']
        # b_l is the b_lock length of compror codewords
        b_l = np.zeros(states - 1)

        h_0 = np.log2(
            np.cumsum([1.0 if sfx == 0 else 0.0 for sfx in self.basic_attributes['sfx'][1:]]))
        h_1 = np.array([np.log2(i + 1) if m == 0 else np.log2(i + 1) + np.log2(m)
                        for i, m in enumerate(self.statistics['max_lrs'][1:])])

        j = 0
        for i, _ in enumerate(code):
            if self.comp_attributes['code'][i][0] == 0:
                b_l[j] = 1
                j += 1
            else:
                loc = code[i][0]
                b_l[j:j + loc] = loc  # range(1,loc+1)
                j = j + loc

        h_1 = h_1 / b_l
        i_r = alpha * h_0 - h_1
        i_r[i_r < 0] = 0  # Really a HACK here!!!!!
        return i_r, h_0, h_1

    def _ir_cum3(self, alpha=1.0):
        """
        Computes a third version of cumulative information rate (IR) based on symbol appearance.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
    
        Returns:
            tuple: A tuple containing the cumulative information rate, h_0, and h_1 values.
        """
        h_0 = np.log2(
            np.cumsum([1.0 if sfx == 0 else 0.0 for sfx in self.basic_attributes['sfx'][1:]]))
        h_1 = np.array([h if m == 0 else (h + np.log2(m)) / m
                        for h, m in zip(h_0, self.basic_attributes['lrs'][1:])])

        i_r = alpha * h_0 - h_1
        i_r[i_r < 0] = 0  # Really a HACK here!!!!!
        return i_r, h_0, h_1

    def i_r(self, alpha=1.0, ir_type='cum'):
        """
        Computes the information rate (IR) based on the specified IR type.
    
        Parameters:
            alpha (float): A scaling factor for adjusting the information rate.
            ir_type (str): The type of IR to compute ('cum', 'all', 'fixed', 'cum2', 'cum3').
    
        Returns:
            tuple: A tuple containing the IR, h_0, and h_1 values.
        """
        if ir_type == 'cum':
            return self._ir_cum(alpha)
        if ir_type == 'all':
            return self._ir(alpha)
        if ir_type == 'fixed':
            return self._ir_fixed(alpha)
        if ir_type == 'cum2':
            return self._ir_cum2(alpha)
        if ir_type == 'cum3':
            return self._ir_cum3(alpha)

        return None

    def num_clusters(self):
        """
        Returns the number of clusters based on reverse suffix links.
    
        Returns:
            int: The number of clusters.
        """
        return len(self.basic_attributes['rsfx'][0])

    def threshold(self):
        """
        Retrieves the threshold value used by the oracle.
    
        Returns:
            int: The threshold value.
    
        Raises:
            ValueError: If the threshold is not set.
        """
        if self.params.get('threshold'):
            return int(self.params.get('threshold'))
        raise ValueError("Threshold is not set!")

    def dfunc(self):
        """
        Retrieves the distance function used by the oracle.
    
        Returns:
            str: The name of the distance function.
    
        Raises:
            ValueError: If the distance function is not set.
        """
        if self.params.get('dfunc'):
            return self.params.get('dfunc')
        raise ValueError("dfunc is not set!")

    def dfunc_handle(self, data, b_vec):
        """
        Applies the distance function handle to the provided data and vector.
    
        Parameters:
            data: The data to compare.
            b_vec: The comparison vector.
    
        Returns:
            The result of the distance function.
    
        Raises:
            ValueError: If the distance function handle is not set.
        """
        if self.params['dfunc_handle']:
            fun = self.params['dfunc_handle']
            return fun(data, b_vec)
        raise ValueError("dfunc_handle is not set!")

    def _len_common_suffix(self, p_1, p_2):
        """
        Computes the length of the common suffix between two states.
    
        Parameters:
            p_1: The first state.
            p_2: The second state.
    
        Returns:
            int: The length of the common suffix.
        """
        if p_2 == self.basic_attributes['sfx'][p_1]:
            return self.basic_attributes['lrs'][p_1]
        while self.basic_attributes['sfx'][p_2] != self.basic_attributes['sfx'][p_1] and p_2 != 0:
            p_2 = self.basic_attributes['sfx'][p_2]
        return min(self.basic_attributes['lrs'][p_1], self.basic_attributes['lrs'][p_2])

    def _find_better(self, i, symbol):
        """
        Finds a better match for a given symbol based on the reverse suffix links.
    
        Parameters:
            i: The index of the current state.
            symbol: The symbol to match.
    
        Returns:
            int or None: The index of the matching state, or None if no match is found.
        """
        self.basic_attributes['rsfx'][self.basic_attributes['sfx'][i]].sort()
        for j in self.basic_attributes['rsfx'][self.basic_attributes['sfx'][i]]:
            new_symbol = self.basic_attributes['data'][j -
                                                       self.basic_attributes['lrs'][i]]
            if (self.basic_attributes['lrs'][j] == self.basic_attributes['lrs'][i]
                    and new_symbol == symbol):
                return j
        return None
