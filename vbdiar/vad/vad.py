#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import logging

import numpy as np


logger = logging.getLogger(__name__)


def get_vad(file_name, fea_len):
    """ Load .lab file as bool vector.

    Args:
        file_name (str): path to .lab file
        fea_len (int): length of features

    Returns:
        np.array: bool vector
    """

    logger.info('Loading VAD from file `{}`.'.format(file_name))
    return load_vad_lab_as_bool_vec(file_name)[:fea_len]


def load_vad_lab_as_bool_vec(lab_file):
    """

    Args:
        lab_file:

    Returns:

    """
    lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

    if lab_cont.shape[1] == 0:
        return np.empty(0), 0, 0

    if lab_cont.shape[1] == 3:
        lab_cont = lab_cont[lab_cont[:, 2] == 'sp', :][:, [0, 1]]

    n_regions = lab_cont.shape[0]
    ii = 0
    while True:
        try:
            start1, end1 = float(lab_cont[ii][0]), float(lab_cont[ii][1])
            jj = ii + 1
            start2, end2 = float(lab_cont[jj][0]), float(lab_cont[jj][1])
            if end1 >= start2:
                lab_cont = np.delete(lab_cont, ii, axis=0)
                ii -= 1
                lab_cont[jj - 1][0] = str(start1)
                lab_cont[jj - 1][1] = str(max(end1, end2))
            ii += 1
        except IndexError:
            break

    vad = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
    vad[1] += 1  # Paja's bug!!!

    if not vad.size:
        return np.empty(0, dtype=bool)

    npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
    npc2 = np.c_[vad[0] - np.r_[0, vad[1, :-1]], vad[1] - vad[0]]
    npc2[npc2 < 0] = 0

    out = np.repeat(npc1, npc2.flat)

    n_frames = sum(out)

    return out, n_regions, n_frames
    

