#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import struct

import numpy as np


WAVEFORM = 0
IREFC = 5
DISCRETE = 10

_E = 0000100  # has energy
_N = 0000200  # absolute energy supressed
_D = 0000400  # has delta coefficients
_A = 0001000  # has acceleration coefficients
_C = 0002000  # is compressed
_Z = 0004000  # has zero mean static coef.
_K = 0010000  # has CRC checksum
_0 = 0020000  # has 0th cepstral coef.
_V = 0040000  # has VQ data
_T = 0100000  # has third differential coef.

parms16bit = [WAVEFORM, IREFC, DISCRETE]


def read_htk(file_name, return_parm_kind_and_samp_period=False):
    """ Read htk feature file
     Input:
         file: file name or file-like object.
     Outputs:
          m  - data: column vector for waveforms, one row per frame for other types
          sampPeriod - frame rate [seconds]
          parmKind
    """
    try:
        fh = open(file_name, 'rb')
    except TypeError:
        fh = file_name
    try:
        n_samples, samp_period, samp_size, parm_kind = struct.unpack(">IIHH", fh.read(12))
        m = np.frombuffer(fh.read(n_samples*samp_size), 'i1')
        pk = parm_kind & 0x3f
        if pk in parms16bit:
            m = m.view('>h').reshape(n_samples, samp_size/2)
        elif parm_kind & _C:
            scale, bias = m[:samp_size*4].view('>f').reshape(2, samp_size/2)
            m = (m.view('>h').reshape(n_samples, samp_size/2)[4:] + bias) / scale
        else:
            m = m.view('>f').reshape(n_samples, samp_size/4)
        if pk == IREFC:
            m = m / 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
        if parm_kind & _K:
            fh.read(1)
    finally:
        if fh is not file_name:
            fh.close()
    return m if not return_parm_kind_and_samp_period else (m, parm_kind, samp_period / 1e7)
