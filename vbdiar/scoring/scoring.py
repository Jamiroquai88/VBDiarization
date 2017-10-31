#!/usr/bin/env python

"""
Code for logistic regression fusion
"""

import numpy as np


################################################################################
################################################################################
def load_gzvectors_into_ndarray(lst, prefix='', suffix='', 
    dtype=np.float64):
    """ Loads the scp list into ndarray
    """
    n_data      = lst.shape[0]
    v_dim       = None

    for ii, segname in enumerate(lst):
        print('Loading [{}/{}] {}'.format(ii, n_data, segname))

        tmp_vec = np.loadtxt(prefix + segname + suffix, dtype=dtype)

        if v_dim == None:
            v_dim   = len(tmp_vec)
            out     = np.zeros((n_data, v_dim), dtype=dtype)
        elif v_dim != len(tmp_vec):
            raise ValueError(str.format("Vector {} is of wrong size ({} instead of {})",
                segname, len(tmp_vec), v_dim))
            
        out[ii,:] = tmp_vec

    return out


def warp2us(ivecs, lda, lda_mu):
    ivecs  = ivecs.dot(lda) - lda_mu
    ivecs /= np.sqrt((ivecs**2).sum(axis=1)[:,np.newaxis])
    return ivecs


def bilinear_plda(Lambda, Gamma, c, k, Fe, Ft, out=None):
    """ Performs a full PLDA scoring
    """
    if out is None:
        out = np.empty((Fe.shape[0], Ft.shape[0]))

    np.dot(Fe.dot(Lambda), Ft.T, out=out)
    out += (np.sum(Fe.dot(Gamma) * Fe, 1) + Fe.dot(c))[:,np.newaxis]
    out += (np.sum(Ft.dot(Gamma) * Ft, 1) + Ft.dot(c))[np.newaxis,:] + k

    return out

