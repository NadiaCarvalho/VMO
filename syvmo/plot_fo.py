#!/usr/bin/env python3.7
"""
This script defines ploting routines for ploting oracles,
based on the code in https://github.com/wangsix/vmo/b_lob/master/vmo/plot.py
"""

import numpy as np

try:
    # , ImageFilter  # @UnresolvedImport @UnusedImport
    from PIL import Image, ImageDraw
except Exception:
    print('pil not loaded - hopefully running in max')


WIDTH = 900 * 4
HEIGHT = 400 * 4
LRS_THRESH = 0


COLOR_SFX = (255, 0, 0, 0)
COLOR_BACKGROUND = (255, 255, 255, 0)
COLOR_TRANS = (0, 0, 153, 0)


def start_draw(_oracle, _offsets=None, size=(900*4, 600*4)):
    """

    :param _oracle: input vmo object
    :param size: the size of the output image
    :return: an update call the draw()
    """

    width = size[0]
    height = size[1]
    image = Image.new('RGB', (width, height), color=COLOR_BACKGROUND)

    if isinstance(_oracle, dict):
        return draw_oracles(oracles=_oracle, offsets=_offsets, current_state=0, image=image, width=width, height=height)
    return draw(oracle=_oracle, current_state=0, image=image, width=width, height=height)


def draw_offsets(oracle, offsets, minor_offset, major_offset, current_state, image, width=WIDTH, height=HEIGHT):
    """
    :param oracle: input vmo object
    :param offsets: input offsets of states of vmo
    :param minor_offset: minimum of the offsets of all oracles
    :param minor_offset: maximum of the offsets of all oracles
    :param current_state:
    :param image: an PIL image object
    :param width: width of the image
    :param height: height of the image
    :return: the updated PIL image object
    """
    trn = oracle.basic_attributes['trn']
    sfx = oracle.basic_attributes['sfx']
    lrs = oracle.basic_attributes['lrs']

    # handle to Draw object - PIL
    n_states = len(sfx)
    drawer = ImageDraw.Draw(image, mode='RGB')

    sum_offset = (major_offset + 2) - minor_offset
    make_offsets_0 = [minor_offset - 0.2] + \
        list(map((lambda x: x + 0.8), offsets))

    make_offsets = list(map((lambda x: x - 0.2), offsets))

    print(offsets)
    print(make_offsets)
    print(make_offsets_0)

    for i in range(n_states):
        # draw circle for each state
        x_pos = (((make_offsets[i] - minor_offset) / sum_offset) *
                 width) + 0.5 * (1.0 / sum_offset) * width
        x_ball = x_pos + (0.25 / sum_offset * width)
        diameter = x_ball - x_pos

        drawer.ellipse([x_pos, height/2 - diameter/2, x_ball,
                        height/2 + diameter/2], outline='green', width=2)

        if i == (n_states-1):
            break

        # iterate over forward transitions
        for tran in trn[i]:
            # if forward transition to next state
            if tran == i + 1:
                # draw forward transitions
                next_x = (((make_offsets[i+1] - minor_offset) / sum_offset) * width) + \
                    0.5 * (1.0 / sum_offset) * width
                current_x = x_pos + (0.25 / sum_offset * width)
                drawer.line((current_x, height/2, next_x, height/2),
                            fill=COLOR_TRANS, width=3)
            else:
                if lrs[tran] >= LRS_THRESH:
                    # forward transition to another state
                    current_x = x_pos
                    next_x = ((float(make_offsets[tran] - minor_offset) / sum_offset)
                              * width) + (0.5 / sum_offset * width)
                    arc_height = (height / 2) + \
                        (make_offsets[tran] - make_offsets[i]) * 0.125
                    drawer.arc((int(current_x) + diameter/2, int(height/2 - arc_height/2) - diameter/2.5,
                                int(next_x) + diameter/2, int(height/2 + arc_height / 2) - diameter/2.5), 180, 0,
                               fill=COLOR_TRANS, width=5)
        if sfx[i] is not None and sfx[i] != 0 and lrs[sfx[i]] >= LRS_THRESH:
            current_x = x_pos
            next_x = (float(make_offsets[sfx[i]] - minor_offset) / sum_offset * width) + \
                (0.5 / sum_offset * width)
            # draw arc
            arc_height = (height / 2) - (make_offsets[sfx[i]] - i) * 0.125
            drawer.arc((int(next_x) + diameter/2,
                        int(height/2 - arc_height/2) + diameter/2.5,
                        int(current_x) + diameter/2,
                        int(height/2 + arc_height/2) + diameter/2.5),
                       0,
                       180,
                       fill=COLOR_SFX,
                       width=5)

    image.resize((900, 600), (Image.BILINEAR))
    return image


def draw_oracles(oracles, offsets, current_state, image, width=WIDTH, height=HEIGHT):
    """
    draw various oracles
    """
    images = []
    max_off = max([offs[-1] for key, offs in offsets.items()])
    min_off = min([offs[0] for key, offs in offsets.items()])
    for key, oracle in oracles.items():
        new_image = Image.new(
            'RGB', (width, int(height/len(oracles))), color=COLOR_BACKGROUND)
        if key in offsets:
            al = draw_offsets(oracle, offsets[key], min_off, max_off, current_state, new_image, width, height=int(
                height/len(oracles)))
        else:
            al = draw(oracle, current_state, new_image,
                      width, height=int(height/len(oracles)))
        images.append(al)

    for i, n_i in enumerate(images):
        image.paste(n_i, (0, int(i*height/len(oracles))))
    return image


def draw_compror():
    """Compror drawing: under construction"""
    raise NotImplementedError(
        "Compror drawing is under construction, coming soon!")


def get_pattern_mat(oracle, pattern):
    """Output a matrix containing patterns in rows from a vmo.

    :param oracle: input vmo object
    :param pattern: pattern extracted from oracle
    :return: a numpy matrix that could be used to visualize the pattern extracted.
    """
    pattern_mat = np.zeros((len(pattern), oracle.statistics['n_states']-1))
    for i, pat in enumerate(pattern):
        length = pat[1]
        for _s in pat[0]:
            pattern_mat[i][_s-length:_s-1] = 1

    return pattern_mat

def find_repeated_patterns(oracle, lower=1):
    """
    Find Repeated Patterns
    """
    if lower < 0:
        lower = 0

    pattern_list = []
    prev_sfx = -1
    for i in range(oracle.statistics['n_states'] - 1, lower + 1, -1):
        # Searching back from the end to the last possible position for repeated patterns
        sfx = oracle.basic_attributes['sfx'][i]
        rsfx = oracle.basic_attributes['rsfx'][i]
        lrs = oracle.basic_attributes['lrs'][i]
        pattern_found = False
        # if (sfx != 0  # not pointing to zeroth state
        #     and i - oracle.basic_attributes['lrs'][i] + 1 > sfx and oracle.basic_attributes['lrs'][i] > lower):  # constraint on length of patterns
        if (sfx != 0  # not pointing to zeroth state
            and lrs > lower):  # constraint on length of patterns
            for p in pattern_list:  # for existing pattern
                if not [_p for _p in p[0] if _p - p[1] < i < _p]:
                    if sfx in p[0]:
                        p[0].append(i)
                        lrs_len = np.min([p[1], lrs])
                        p[1] = lrs_len
                        pattern_found = True
                        break
                    else:
                        pattern_found = False

            if prev_sfx - sfx != 1 and not pattern_found:
                _rsfx = np.array(rsfx).tolist()
                if _rsfx:
                    _rsfx.extend([i, sfx])
                    _len = np.array(oracle.basic_attributes['lrs'])[_rsfx[:-1]].min()
                    if i - _len + 1 < sfx:
                        _len = i-sfx
                    if _len > lower:
                        pattern_list.append([_rsfx, _len])
                else:
                    if i - lrs + 1 < sfx:
                        pattern_list.append([[i, sfx], i-sfx])
                    else:
                        pattern_list.append([[i, sfx], lrs])
            prev_sfx = sfx
        else:
            prev_sfx = -1
    return pattern_list

def find_fragments(oracle):
    seg_list = []
    seg_rsfx = []
    pos = oracle.statistics['n_states'] - 1
    while pos > 0:
        lrs = oracle.basic_attributes['lrs'][pos]
        rsfx = oracle.basic_attributes['rsfx'][pos]
        if lrs > 1:
            _lrs_of_seg = np.array(oracle.basic_attributes['lrs'][pos - lrs + 1:pos])

            if lrs < np.max(_lrs_of_seg):
                stop = np.where(lrs < _lrs_of_seg)[0][-1]
                lrs = lrs - stop - 1
            seg = [pos, lrs]
            seg_list.append(seg)
            pos -= lrs
        else:
            seg_list.append([pos, 1])
            pos -= 1
        if rsfx and np.min(rsfx) in [s[0] for s in seg_list]:
            seg_rsfx.insert(0, [s[0] for s in seg_list].index(np.min(rsfx)))
        else:
            seg_rsfx.insert(0, 0)

    for i, r in enumerate(seg_rsfx):
        if r > 0:
            seg_rsfx[i] += i

    return seg_list, seg_rsfx


def draw(oracle, current_state, image, width=WIDTH, height=HEIGHT):
    """

    :param oracle: input vmo object
    :param current_state:
    :param image: an PIL image object
    :param width: width of the image
    :param height: height of the image
    :return: the updated PIL image object
    """

    trn = oracle.basic_attributes['trn']
    sfx = oracle.basic_attributes['sfx']
    lrs = oracle.basic_attributes['lrs']

    # handle to Draw object - PIL
    n_states = len(sfx)
    drawer = ImageDraw.Draw(image, mode='RGB')

    for i in range(n_states):
        # draw circle for each state
        x_pos = (float(i) / n_states * width) + 0.5 * 1.0 / n_states * width
        x_ball = x_pos + (0.25 / n_states * width)
        diameter = x_ball - x_pos

        drawer.ellipse([x_pos, height/2 - diameter/2, x_ball,
                        height/2 + diameter/2], outline='green', width=3)

        # iterate over forward transitions
        for tran in trn[i]:
            # if forward transition to next state
            if tran == i + 1:
                # draw forward transitions
                next_x = (float(i + 1) / n_states * width) + \
                    0.5 * 1.0 / n_states * width
                current_x = x_pos + (0.25 / n_states * width)
                drawer.line((current_x, height/2, next_x, height/2),
                            width=3, fill=COLOR_TRANS)
            else:
                if lrs[tran] >= LRS_THRESH:
                    # forward transition to another state
                    current_x = x_pos
                    next_x = (float(tran) / n_states * width) + \
                        (0.5 / n_states * width)
                    arc_height = (height / 2) + (tran - i) * 0.125
                    drawer.arc((int(current_x) + diameter/2, int(height/2 - arc_height/2) - diameter/2.5,
                                int(next_x) + diameter/2, int(height/2 + arc_height / 2) - diameter/2.5), 180, 0,
                               fill=COLOR_TRANS, width=5)
        if sfx[i] is not None and sfx[i] != 0 and lrs[sfx[i]] >= LRS_THRESH:
            current_x = x_pos
            next_x = (float(sfx[i]) / n_states * width) + \
                (0.5 / n_states * width)
            # draw arc
            arc_height = (height / 2) - (sfx[i] - i) * 0.125
            drawer.arc((int(next_x) + diameter/2,
                        int(height/2 - arc_height/2) + diameter/2.5,
                        int(current_x) + diameter/2,
                        int(height/2 + arc_height/2) + diameter/2.5),
                       0,
                       180,
                       fill=COLOR_SFX, width=5)

    image.resize((450, 200), (Image.BILINEAR))
    return image
