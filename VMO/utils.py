#!/usr/bin/env python3.7
"""
This script defines the utils for the FO family algorithms
based on the code in https://github.com/wangsix/vmo/blob/master/vmo/VMO/oracle.py
"""

import numpy as np
import time

from VMO.oracles.fo import FO
from VMO.oracles.vmo import VMO


def _create_oracle(oracle_type, **kwargs):
    """A routine for creating a factor oracle."""
    if oracle_type == 'f':
        return FO(**kwargs)
    return VMO(**kwargs)


def create_oracle(flag, threshold=0, dfunc='euclidean',
                  dfunc_handle=None, dim=1, weights=None,
                  fixed_weights=None):
    """docstring"""
    return _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                          dfunc_handle=dfunc_handle, dim=dim, weights=weights,
                          fixed_weights=fixed_weights)


def _build_oracle(flag, oracle, input_data, suffix_method='inc'):
    """A routine for building a factor oracle."""

    if not isinstance(input_data, np.ndarray) or not isinstance(input_data[0], np.ndarray):
        input_data = np.array(input_data)

    if input_data.ndim != 2:
        input_data = np.expand_dims(input_data, axis=1)

    if flag == 'a':
        _ = [oracle.add_state(obs, suffix_method) for obs in input_data]
        oracle.f_array.finalize()
    else:
        _ = [oracle.add_state(obs) for obs in input_data]
    return oracle


def build_oracle(input_data, flag,
                 threshold=0, suffix_method='inc',
                 features=None, weights=None, fixed_weights=None, dfunc='cosine',
                 dfunc_handle=None, dim=1):
    """docstring"""
    # initialize weights if needed
    if weights is None:
        if features is not None:
            weights = np.array([1.0 for feature in features])

    if 'f' or 'v' in flag:
        oracle = _create_oracle(flag, threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim, weights=weights,
                                fixed_weights=fixed_weights)
        oracle = _build_oracle(flag, oracle, input_data)
    else:
        oracle = _create_oracle('a', threshold=threshold, dfunc=dfunc,
                                dfunc_handle=dfunc_handle, dim=dim, weights=weights,
                                fixed_weights=fixed_weights)
        oracle = _build_oracle(flag, oracle, input_data, suffix_method)

    return oracle


def find_threshold(input_data, _r=(0, 1, 0.1), method='ir', flag='a',
                   suffix_method='inc', alpha=1.0, features=None, weights=None,
                   fixed_weights=None, ir_type='cum',
                   dfunc='cosine', dfunc_handle=None, dim=1,
                   verbose=False, entropy=False):
    """docstring"""
    if method == 'ir':
        return find_threshold_ir(input_data, _r, flag, suffix_method, alpha,
                                 features, weights, fixed_weights, ir_type, dfunc, dfunc_handle, dim,
                                 verbose, entropy)
    return None


def find_threshold_ir(input_data, _r=(0, 1, 0.1), flag='a', suffix_method='inc',
                      alpha=1.0, features=None, weights=None, fixed_weights=None,
                      ir_type='cum', dfunc='cosine', dfunc_handle=None, dim=1,
                      verbose=False, entropy=False):
    """docstring"""
    thresholds = np.arange(_r[0], _r[1], _r[2])
    irs = []

    if entropy:
        h0_vec = []
        h1_vec = []

    for _t in thresholds:
        if verbose:
            print('Testing threshold:', _t)
        start_time = time.time()
        tmp_oracle = build_oracle(input_data, flag=flag, threshold=_t,
                                  suffix_method=suffix_method, features=features,
                                  dfunc=dfunc, dfunc_handle=dfunc_handle, dim=dim,
                                  weights=weights, fixed_weights=fixed_weights)
        tmp_ir, h_0, h_1 = tmp_oracle.i_r(ir_type=ir_type, alpha=alpha)
        irs.append(tmp_ir.sum())
        if entropy:
            h0_vec.append(h_0.sum())
            h1_vec.append(h_1.sum())
        if verbose:
            print("--- %s seconds ---\n" % (time.time() - start_time))

    # now pair irs and thresholds in a vector, and sort by ir
    ir_thresh_pairs = list(zip(irs, thresholds))
    pairs_return = ir_thresh_pairs
    ir_thresh_pairs = sorted(ir_thresh_pairs, key=lambda x: x[0],
                             reverse=True)

    if entropy:
        return ir_thresh_pairs[0], pairs_return, h0_vec, h1_vec
    return ir_thresh_pairs[0], pairs_return
