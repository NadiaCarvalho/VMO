#!/usr/bin/env python3.7
"""
This script defines the generation algorithm for synchronization of various Oracles
"""

import random

import numpy as np

from ..oracles.factor_oracle import FactorOracle


def sync_generate(oracles, offsets, seq_len=10, p=0.5, k=1):
    """
    Generate synchronized lines from various oracles
    """
    len_events_by_part = [(key, len(part[:-1])) for key, part in offsets.items()]
    len_events_by_part.sort(key=lambda tup: tup[1], reverse=True)
    principal_key = len_events_by_part[0][0]
    max_size = len_events_by_part[0][1] + 1

    trns = {}
    sfxs = {}
    lrss = {}
    rsfxs = {}
    sequences = {}
    ktraces = {}

    if k == -1:
        if not all(offsets[key][-1] == offsets[principal_key][-1] for key in offsets.keys()):
            reversed_offsets = list(reversed(offsets[principal_key]))
            for _j, off in enumerate(reversed_offsets):
                new_k = len(reversed_offsets) - _j - 1
                ks_at_off = _find_ks(offsets, principal_key, new_k)
                if len(ks_at_off.keys()) == len(offsets.keys()):
                    k = ks_at_off[principal_key] - 1
                    break

    offsets_at_k = _find_ks(offsets, principal_key, k)
    for key, oracle in oracles.items():
        trns[key] = oracle.basic_attributes['trn'][:]
        sfxs[key] = oracle.basic_attributes['sfx'][:]
        lrss[key] = oracle.basic_attributes['lrs'][:]
        rsfxs[key] = oracle.basic_attributes['rsfx'][:]
        sequences[key] = []

        if key in offsets_at_k:
            ktraces[key] = [offsets_at_k[key]]
        else:
            ktraces[key] = [-1]

    times = 0
    for _i in range(seq_len):
        print(k)

        ks_at_k = _find_ks(offsets, principal_key, k)
        sfxs_k = dict([(key, sfxs[key][ks_at_k[key]])
                       for key in ks_at_k.keys()])

        # generate each state
        if _i == 0 and k != -1 and all(ks < len(sfxs[key]) - 1 for key, ks in ks_at_k.items()):
            print('INITIAL STRAIGHT FORWARD')
            key = principal_key
            sym = k + 1
        elif any(list(sfxs_k.values())):
            if (random.random() < p):
                print('TRANSITIONS')
                key, sym = copy_transitions(
                    k, trns, sfxs, ks_at_k, ktraces, offsets, principal_key)
            else:
                print('SFXS BEST LRSS')
                key, sym = jump_best_lrss(
                    sfxs, rsfxs, lrss, max_size, ktraces, ks_at_k, offsets, principal_key)
                sym += 1
        else:
            if all(ks < len(sfxs[key]) - 1 for key, ks in ks_at_k.items()):
                print('STRAIGHT FORWARD')
                key = principal_key
                sym = k + 1
            else:
                print('SFXS RANDOM')
                key, sym = choose_from_sfxs_k(
                    k, sfxs, ks_at_k, offsets, principal_key)
                sym += 1

        next_keys = _find_ks(offsets, key, sym)
        last_ktraces = dict([(key, ktr[-1]) for key, ktr in ktraces.items()])
        if last_ktraces[key] == sym and times > 3:
            key, sym = choose_from_sfxs_k(
                k, sfxs, ks_at_k, offsets, principal_key)
            if sym is not None:
                sym += 1
            else:
                sym = 0
        if last_ktraces[key] == sym:
            times += 1
        else:
            times = 0

        offsets_at_kp = dict([(key, offsets[key][ks-1]) if ks < len(
            offsets[key]) and ks > 1 else (key, -1) for key, ks in next_keys.items()])
        start_offset = 0
        if len(offsets_at_kp.values()) > 0 and principal_key in offsets_at_kp:
            start_offset = offsets_at_kp[principal_key]
        keys_at_beginning_of_event = dict([(key, _find_nearest_k(
            offsets, key, start_offset)) for key in last_ktraces.keys()])

        offsets_at_kp1 = dict([(key, offsets[key][ks]) if ks < len(
            offsets[key]) else (key, -1) for key, ks in next_keys.items()])

        max_offset = -1
        if len(offsets_at_kp1.values()) > 0:
            max_offset = max(offsets_at_kp1.values())
        if any(off == -1 for off in offsets_at_kp1.values()):
            max_offset = -1
        ks_at_final_of_event = dict(
            [(key, _find_nearest_k(offsets, key, max_offset, minus=False) + 1) for key in sfxs.keys()])

        for key in sfxs.keys():
            if key in next_keys:
                if _i != 0:
                    sequences[key].append(next_keys[key])
                ktraces[key].append(next_keys[key])
                sequences, ktraces = fill_gaps(
                    key, ks_at_final_of_event, next_keys[key], sequences, ktraces, start_offset, offsets, max_offset, _i=_i)
            else:
                sequences, ktraces = fill_gaps(
                    key, ks_at_final_of_event, keys_at_beginning_of_event[key],
                    sequences, ktraces, start_offset, offsets, max_offset, no_key=True, _i=_i)

        k = ktraces[principal_key][-1]
        if any(ktr[-1] >= len(sfxs[key]) - 1 for key, ktr in ktraces.items()):
            k = 0

        print(ktraces)
        print(sequences)

    print('____________________________________________________________________')

    return sequences, ktraces


def fill_gaps(key, ks_dict_1, k_2, sequences, ktraces, start_offset, offsets, max_offset, no_key=False, _i=0):
    """
    Fill Gaps where existent
    """
    max_of_all_offsets = max(max(offsets.values())) + 1
    values = ks_dict_1[key] - k_2
    if max_offset == -1:
        max_offset = max(max(offsets.values())) + 1

    for i in range(values):
        ks = k_2 + i + 1
        if ks == len(offsets[key]) and max_offset == max_of_all_offsets:
            if _i != 0:
                sequences[key].append(ks)
            ktraces[key].append(ks)
        elif ks < len(offsets[key]):
            if  _i != 0 and i == 0 and offsets[key][ks - 1] > start_offset and no_key:
                duration = str(abs(start_offset - offsets[key][ks  - 1]))
                sequences[key].append('N_' + duration)
            elif offsets[key][ks - 1] >= start_offset and offsets[key][ks] <= max_offset:
                if _i != 0:
                    sequences[key].append(ks)
                ktraces[key].append(ks)

            if _i != 0 and (i == values - 1) and offsets[key][ks - 1] < max_offset:
                duration = str(abs(max_offset - offsets[key][ks]))
                sequences[key].append('N_' + duration)

    if  _i != 0 and no_key and values <= 0 and k_2 < len(offsets[key]):
        duration_minor = str(abs(start_offset - offsets[key][k_2 - 1]))
        duration_max = str(abs(max_offset - offsets[key][k_2]))
        sequences[key].append('N_' + min(duration_minor, duration_max))

    return sequences, ktraces


def choose_from_sfxs_k(k, sfxs, ks_at_k, offsets, principal_key):
    """
    Choose a random state from sfxs at k
    """
    sfxs_k = dict([(key, sfxs[key][ks_at_k[key]])
                   for key in ks_at_k.keys()])
    possible_moves = [(key, sfx) for key, sfx in sfxs_k.items() if len(
        _find_ks(offsets, key, sfx).values()) == len(offsets.keys())]

    pr_key = principal_key
    sym = sfxs_k[principal_key]
    if len(possible_moves) > 0:
        pr_key, sym = possible_moves[int(
            np.floor(random.random() * len(possible_moves)))]
    return pr_key, sym


def copy_transitions(k, trns, sfxs, ks_at_k, ktraces, offsets, principal_key, i=0):
    """
    Copy forward according to possible transitions for all oracles
    """
    I = get_f_transitions_by_oracle(
        trns, _find_ks(offsets, principal_key, k), offsets)

    if i > 5:
        key, sym = choose_from_sfxs_k(k, sfxs, ks_at_k, offsets, principal_key)
        sym += 1
        return key, sym

    if len(I) == 0:
        key_pr, sym = choose_from_sfxs_k(
            k, sfxs, ks_at_k, offsets, principal_key)
        new_ks = _find_ks(offsets, key_pr, sym)
        _ = [ktraces[key].append(value) for key, value in new_ks.items()]
        return copy_transitions(new_ks[principal_key], trns, sfxs, ks_at_k, ktraces, offsets, principal_key, i+1)

    return I[int(np.floor(random.random() * len(I)))]


def jump_best_lrss(sfxs, rsfxs, lrss, max_size, ktraces, ks_at_k, offsets, principal_key):
    """
    Get Best Suffix to Jump
    """
    _ = [ktraces[key].append(value) for key, value in ks_at_k.items()]
    key, sym = get_next_suffix(
        sfxs, rsfxs, lrss, max_size, ks_at_k, offsets)
    if key == '':
        key = principal_key
    return key, sym


def get_sim_trans(I, offsets, key):
    """
    Get Simultaneous Transitions by K
    """
    return [trans for trans in I if len(_find_ks(offsets, key, trans).values()) == len(offsets.keys())]


def get_f_transitions_by_oracle(trns, ks_at_k, offsets):
    """
    Get all Possible Transitions at offset k
    """
    trans = dict([(key, trns[key][ks_at_k[key]])
                  for key in ks_at_k.keys()])
    return [(key, tr) for key, I in trans.items() for tr in get_sim_trans(I, offsets, key)]


def get_next_suffix(sfxs, rsfxs, lrss, max_size, ks_at_k, offsets):
    """
    Try candidate suffix links for all oracles,
    find the one that gets the maximum lrs position
    and return
    """
    k_vecs = dict([(key, []) for key in sfxs.keys()])
    k_vecs = dict([(key, _find_links(k_vecs[key], sfxs[key], rsfxs[key],
                                     ks_at_k[key])) for key in ks_at_k.keys()])
    lrs_vecs = dict([(key, [lrss[key][_i] for _i in k_vec])
                     for key, k_vec in k_vecs.items()])

    k_vecs, lrs_vecs = filter_lrss(k_vecs, lrs_vecs, offsets)
    return get_max_lrs_position(k_vecs, lrs_vecs, max_size)


def filter_lrss(k_vecs, lrs_vecs, offsets):
    """
    Filter LRSs that go to a state that has
    a next state with events in all voices
    """
    k_vecs_filtered = {}
    lrs_vecs_filtered = {}
    for key, k_poss in lrs_vecs.items():
        k_vecs_filtered[key] = []
        lrs_vecs_filtered[key] = []

        for i, k in enumerate(k_poss):
            if len(_find_ks(offsets, key, k + 1).values()) == len(offsets.keys()):
                k_vecs_filtered[key].append(k_vecs[key][i])
                lrs_vecs_filtered[key].append(k)

        if len(k_vecs_filtered[key]) == 0:
            k_vecs_filtered[key] = k_vecs[key]
            lrs_vecs_filtered[key] = lrs_vecs_filtered[key]

    return k_vecs_filtered, lrs_vecs_filtered


def get_max_lrs_position(k_vecs, lrs_vecs, length):
    """
    Find the position of the max lrs
    """
    key_lrss = [''] * length
    max_lrss = [0] * length

    for key, k_vec in k_vecs.items():
        for i, k in enumerate(k_vec):
            if lrs_vecs[key][i] >= max_lrss[k]:
                max_lrss[k] = lrs_vecs[key][i]
                key_lrss[k] = key

    max_lrs_value = max(max_lrss)
    indexes_list = [i for i in range(len(max_lrss)) if max_lrss[i] == max_lrs_value]
    print(indexes_list)
    max_lrs = indexes_list[int(
            np.floor(random.random() * len(indexes_list)))]

    return key_lrss[max_lrs], max_lrs


def _find_ks(offsets, principal_key, k):
    """
    Find k for parts at offset x
    """
    if k == 0:
        return dict([(key, 0) for key in offsets.keys()])

    result = dict([(key, off.index(offsets[principal_key][k-1])+1)
                   for key, off in offsets.items() if k is not None and k-1 < len(offsets[principal_key]) and offsets[principal_key][k-1] in off])
    return result


def _find_nearest_k(offsets, key, off, minus=True):
    """Find nearest k to an offset"""
    if off == -1:
        return len(offsets[key])

    k = np.argmin(np.abs(np.array(offsets[key])-off))
    if offsets[key][k] <= off and minus:
        k += 1
    elif offsets[key][k] > off and not minus:
        k -= 1

    return k


def _find_links(k_vec, sfx, rsfx, k):
    """Find sfx/rsfx recursively."""
    k_vec.sort()
    if 0 in k_vec:
        return k_vec
    else:
        if sfx[k] not in k_vec:
            k_vec.append(sfx[k])
        for i in range(len(rsfx[k])):
            if rsfx[k][i] not in k_vec:
                k_vec.append(rsfx[k][i])
        for i in range(len(k_vec)):
            k_vec = _find_links(k_vec, sfx, rsfx, k_vec[i])
            if 0 in k_vec:
                break
        return k_vec
