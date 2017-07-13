#!/usr/bin/env python

import sys
import numpy as np
import ivector_io as ivio



################################################################################
################################################################################
def load_gzvectors_into_ndarray(lst, prefix='', suffix='', dtype=np.float64):
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


################################################################################
################################################################################
def load_vectors_into_ndarray(lst, prefix='', suffix='', dtype=np.float64):
    """ Loads the scp list into ndarray
    """
    n_data      = lst.shape[0]
    v_dim       = None

    for ii, segname in enumerate(lst):
        print('Loading [{}/{}] {}'.format(ii, n_data, segname))

        tmp_vec, n_frames, tags = ivio.read_binary_ivector(prefix + segname + suffix)

        if v_dim == None:
            v_dim   = len(tmp_vec)
            out     = np.zeros((n_data, v_dim), dtype=dtype)
        elif v_dim != len(tmp_vec):
            raise ValueError(str.format("Vector {} is of wrong size ({} instead of {})",
                segname, len(tmp_vec), v_dim))
            
        out[ii,:] = tmp_vec

    return out


################################################################################
################################################################################
def warp2us(ivecs, lda, lda_mu):
    """ i-vector pre-processing
        This function applies a global LDA, mean subtraction, and length 
        normalization.
    """
    ivecs  = ivecs.dot(lda) - lda_mu
    ivecs /= np.sqrt((ivecs**2).sum(axis=1)[:,np.newaxis])
    return ivecs


################################################################################
################################################################################
def bilinear_plda(Lambda, Gamma, c, k, Fe, Ft):
    """ Performs a full PLDA scoring
    """
    out = np.empty((Fe.shape[0], Ft.shape[0]), dtype=Lambda.dtype)

    np.dot(Fe.dot(Lambda), Ft.T, out=out)
    out += (np.sum(Fe.dot(Gamma) * Fe, 1) + Fe.dot(c))[:,np.newaxis]
    out += (np.sum(Ft.dot(Gamma) * Ft, 1) + Ft.dot(c))[np.newaxis,:] + k

    return out


################################################################################
################################################################################

enroll_scp_list = sys.argv[1]
enroll_dir      = sys.argv[2]
test_scp_list   = sys.argv[3]
test_dir        = sys.argv[4]
plda_model_dir  = sys.argv[5]      
out_file        = sys.argv[6]


################################################################################
################################################################################
print 'Loading backend model from ' + plda_model_dir

lda_file    = plda_model_dir + '/backend.LDA.txt.gz'
mu_file     = plda_model_dir + '/backend.mu_train.txt.gz'
Gamma_file  = plda_model_dir + '/backend.PLDA.Gamma.txt.gz'
Lambda_file = plda_model_dir + '/backend.PLDA.Lambda.txt.gz'
c_file      = plda_model_dir + '/backend.PLDA.c.txt.gz'
k_file      = plda_model_dir + '/backend.PLDA.k.txt.gz'


lda         = np.loadtxt(lda_file,    dtype=np.float32)
mu          = np.loadtxt(mu_file,     dtype=np.float32)
Gamma       = np.loadtxt(Gamma_file,  dtype=np.float32)
Lambda      = np.loadtxt(Lambda_file, dtype=np.float32)
c           = np.loadtxt(c_file,      dtype=np.float32)
k           = np.loadtxt(k_file,      dtype=np.float32)


enroll_prefix = enroll_dir + "/"
enroll_suffix = ".ivec"
test_prefix   = test_dir + "/"
test_suffix   = ".ivec"


print 'Loading list of enroll ivectors from ' + enroll_scp_list
enroll_seg_list = np.atleast_1d(np.loadtxt(enroll_scp_list, dtype=object))

print 'Loading enroll ivectors ' + enroll_scp_list
enroll_ivec  = load_vectors_into_ndarray(enroll_seg_list, prefix=enroll_prefix, suffix=enroll_suffix, dtype=np.float32)


print 'Loading list of test ivectors from ' + test_scp_list
test_seg_list = np.atleast_1d(np.loadtxt(test_scp_list, dtype=object))

print 'Loading test ivectors ' + test_scp_list
test_ivec  = load_vectors_into_ndarray(test_seg_list, prefix=test_prefix, suffix=test_suffix, dtype=np.float32)


print 'Transforming and normalizing i-vectors'
enroll_ivec = warp2us(enroll_ivec, lda, mu)
test_ivec   = warp2us(test_ivec, lda, mu)

print 'Computing PLDA score'
s = bilinear_plda(Lambda, Gamma, c, k, enroll_ivec, test_ivec)

print 'Saving score matrix to ' + out_file
np.savetxt(out_file, s, fmt='%f')



