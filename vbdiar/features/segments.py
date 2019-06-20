#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import logging

import numpy as np


logger = logging.getLogger(__name__)

RATE = 16000
SOURCERATE = 1250
TARGETRATE = 100000

ZMEANSOURCE = True
WINDOWSIZE = 250000.0
USEHAMMING = True
PREEMCOEF = 0.97
NUMCHANS = 24
CEPLIFTER = 22
NUMCEPS = 19
ADDDITHER = 1.0
RAWENERGY = True
ENORMALISE = True

deltawindow = accwindow = 2

cmvn_lc = 150
cmvn_rc = 150

fs = 1e7 / SOURCERATE


def get_segments(vad, max_size, tolerance):
    """ Return clustered speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered segments
        :rtype: list
    """
    clusters = get_clusters(vad, tolerance)
    segments = []
    max_frames = get_frames_from_time(max_size)
    for item in clusters.values():
        if item[1] - item[0] > max_frames:
            for ss in split_segment(item, max_frames):
                segments.append(ss)
        else:
            segments.append(item)
    return segments


def split_segment(segment, max_size):
    """ Split segment to more with adaptive size.

        :param segment: input segment
        :type segment: tuple
        :param max_size: maximal size of window in ms
        :type max_size: int
        :returns: splitted segment
        :rtype: list
    """
    size = segment[1] - segment[0]
    num_segments = int(np.math.ceil(size / max_size))
    size_segment = size / num_segments
    for ii in range(num_segments):
        yield (int(segment[0] + ii * size_segment), int(segment[0] + (ii + 1) * size_segment))


def get_frames_from_time(n):
    """ Get number of frames from ms.

        :param n: number of ms
        :type n: int
        :returns: number of frames
        :rtype: int

        >>> get_frames_from_time(25)
        1
        >>> get_frames_from_time(35)
        2
    """
    assert n >= 0, 'Time must be at least equal to 0.'
    if n < 25:
        return 0
    return int(1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000))


def get_time_from_frames(n):
    """ Get count of ms from number of frames.

        :param n: number of frames
        :type n: int
        :returns: number of ms
        :rtype: int

        >>> get_time_from_frames(1)
        25
        >>> get_time_from_frames(2)
        35

    """
    return int(n * (TARGETRATE / 10000) - (TARGETRATE / 10000) + (WINDOWSIZE / 10000))


def get_clusters(vad, tolerance=10):
    """ Cluster speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered speech segments
        :rtype: dict
    """
    num_prev = 0
    in_tolerance = 0
    num_clusters = 0
    clusters = {}
    for ii, frame in enumerate(vad):
        if frame:
            num_prev += 1
        else:
            in_tolerance += 1
            if in_tolerance > tolerance:
                if num_prev > 0:
                    clusters[num_clusters] = (ii - num_prev, ii)
                    num_clusters += 1
                num_prev = 0
                in_tolerance = 0
    if num_prev > 0:
        clusters[num_clusters] = (ii - num_prev, ii)
        num_clusters += 1
    return clusters


def split_seq(seq, size):
    """ Split up seq in pieces of size.

    Args:
        seq:
        size:

    Returns:

    """
    return [seq[i:i + size] for i in range(0, len(seq), size)]
