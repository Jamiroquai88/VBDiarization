#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved
import h5py
import scipy
import struct
import scipy.linalg
import numpy as np


class GMM(object):

    def __init__(self, model_path):
        """ Initialize Gaussian mixture model.

        Args:
            model_path (str): path to GMM model
        """
        self.ubm_weights, self.ubm_means, self.ubm_variances = GMM.load(model_path)
        self.num_g, self.dim_f = self.ubm_means.shape[0], self.ubm_means.shape[1]
        if self.ubm_variances.size != np.count_nonzero(self.ubm_variances):
            raise ValueError('UBM GMM model contains zeros. Avoiding division by zero.')
        self.ubm_norm_variances = 1 / np.sqrt(self.ubm_variances)
        self.ubm_gmm = GMM.gmm_eval_prep(self.ubm_weights, self.ubm_means, self.ubm_variances)

    def normalize_stats(self, n, ff):
        # TODO find out which variable does what
        """ Normalize UBM statistics.

        Args:
            n:
            ff:

        Returns:

        """
        f0 = ff - self.ubm_means * np.kron(np.ones((self.dim_f, 1), dtype=n.dtype), n).transpose()
        if self.ubm_norm_variances.ndim == 2:
            f0 = f0 * self.ubm_norm_variances
        else:
            for ii in range(self.num_g):
                f0[ii, :] = f0[ii, :].dot(self.ubm_norm_variances[ii])
        return n, f0

    @staticmethod
    def load(fname):
        """ Load GMM UBM in h5 format.

        Args:
            fname (path): path to file in h5 format

        Returns:
            tuple: weights, means, covariances
        """
        model = h5py.File(fname, 'r')
        return model['weights'][:], model['means'][:], model['covs'][:]

    @staticmethod
    def gmm_eval_prep(weights, means, covs):
        """ Prepare GMM for evaluation of statistics.

        Args:
            weights (np.array): GMM weights
            means (np.array): GMM means
            covs (np.array): GMM covariances

        Returns:
            dict: processed GMM dictionary
        """
        n_gauss_mix, dim = means.shape
        gmm = dict()
        is_full_cov = covs.shape[1] != dim
        gmm['utr'], gmm['utc'] = GMM.uppertri_indices(dim, not is_full_cov)

        if is_full_cov:
            gmm['gconsts'] = np.zeros(n_gauss_mix)
            gmm['gconsts2'] = np.zeros(n_gauss_mix)
            gmm['invCovs'] = np.zeros_like(covs)
            gmm['invCovMeans'] = np.zeros_like(means)

            for ii in xrange(n_gauss_mix):
                GMM.uppertri1d_to_sym(covs[ii], gmm['utr'], gmm['utc'])

                inv_c, logdet_c = GMM.inv_posdef_and_logdet(GMM.uppertri1d_to_sym(covs[ii], gmm['utr'], gmm['utc']))

                # log of Gauss. dist. normalizer + log weight + mu' invCovs mu
                inv_cov_mean = inv_c.dot(means[ii])
                gmm['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (
                                            logdet_c + means[ii].dot(inv_cov_mean) + dim * np.log(2.0 * np.pi))
                gmm['gconsts2'][ii] = - 0.5 * (logdet_c + means[ii].dot(inv_cov_mean) + dim * np.log(2.0 * np.pi))
                gmm['invCovMeans'][ii] = inv_cov_mean

                # Iverse covariance matrices are stored in columns of 2D matrix as vectorized upper triangual parts ...
                gmm['invCovs'][ii] = GMM.uppertri1d_from_sym(inv_c, gmm['utr'], gmm['utc'])
            # ... with elements above the diagonal multiply by 2
            gmm['invCovs'][:, dim:] *= 2.0
        else:  # for diagonal
            if covs.size != np.count_nonzero(covs):
                raise ValueError('cov contains zeros. Avoiding division by zero.')
            gmm['invCovs'] = 1 / covs
            gmm['gconsts'] = np.log(weights) - 0.5 * (np.sum(np.log(covs) + means ** 2 * gmm['invCovs'], axis=1)
                                                      + dim * np.log(2.0 * np.pi))
            gmm['gconsts2'] = - 0.5 * (np.sum(np.log(covs) + means ** 2 * gmm['invCovs'], axis=1)
                                       + dim * np.log(2.0 * np.pi))
            gmm['invCovMeans'] = gmm['invCovs'] * means

        # for weight = 0, prepare GMM for uninitialized model with single gaussian
        if len(weights) == 1 and weights[0] == 0:
            gmm['invCovs'] = np.zeros_like(gmm['invCovs'])
            gmm['invCovMeans'] = np.zeros_like(gmm['invCovMeans'])
            gmm['gconsts'] = np.ones(1)
        return gmm

    @staticmethod
    def uppertri1d_from_sym(cov_full, utr, utc):
        # TODO rename method, find out why it is needed
        """ Return dictionary value by key from two values.

        Args:
            cov_full (dict): input dictionary
            utr: first value
            utc: second value

        Returns:

        """
        return cov_full[(utr, utc)]

    @staticmethod
    def inv_posdef_and_logdet(m):
        # TODO find out what it does
        """

        Args:
            m:

        Returns:

        """
        u = np.linalg.cholesky(m)
        logdet = 2 * np.sum(np.log(np.diagonal(u)))
        inv_m = scipy.linalg.solve(m, np.identity(m.shape[0], m.dtype), sym_pos=True)
        return inv_m, logdet

    @staticmethod
    def uppertri1d_to_sym(covs_ut1d, utr, utc):
        """

        Args:
            covs_ut1d:
            utr:
            utc:

        Returns:

        """
        return GMM.uppertri_to_sym(np.array(covs_ut1d)[:, None], utr, utc)[:, :, 0]

    @staticmethod
    def uppertri_to_sym(covs_ut2d, utr, utc):
        # TODO add description of args
        """ Reformat vectorized upper triangual matrices efficiently stored in columns of 2D matrix into
            full symmetric matrices stored in 3rd dimension of 3D matrix.

            Args:
                covs_ut2d:
                utr:
                utc:

            Returns:

        """
        ut_dim, n_mix = covs_ut2d.shape
        dim = (np.sqrt(1 + 8 * ut_dim) - 1) / 2

        covs_full = np.zeros((dim, dim, n_mix), dtype=covs_ut2d.dtype)
        for ii in xrange(n_mix):
            covs_full[:, :, ii][(utr, utc)] = covs_ut2d[:, ii]
            covs_full[:, :, ii][(utc, utr)] = covs_ut2d[:, ii]
        return covs_full

    @staticmethod
    def uppertri_indices(dim, isdiag=False):
        """ Returns row and column indices into upper triangular part of DxD matrices.
            Indices go in zigzag feshinon starting by diagonal. For convenient encoding of
            diagonal matrices, 1:D ranges are returned for both outputs utr and utc when ISDIAG is true.

            Args:
                dim:
                isdiag:

            Returns:

        """
        if isdiag:
            utr = np.arange(dim)
            utc = np.arange(dim)
        else:
            utr = np.hstack([np.arange(ii) for ii in range(dim, 0, -1)])
            utc = np.hstack([np.arange(ii, dim) for ii in range(dim)])
        return utr, utc

    @staticmethod
    def gmm_eval(data, gmm, return_accums=0):
        """ Returns vector of log-likelihoods evaluated for each
            frame of dimXn_samples data matrix using GMM object. GMM object must be
            initialized with GMM_EVAL_PREP function.

            [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistic.

            [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistic.
            For full covariance model second order statiscics, only the vectorized upper
            triangual parts are stored in columns of 2D matrix (similarly to GMM.invCovs).

            Args:
                data (np.array): input data to evaluate
                gmm (np.array): input UBM GMM model
                return_accums (int):

            Returns:
                tuple: statistics
        """
        # quadratic expansion of data
        data_sqr = data[:, gmm['utr']] * data[:, gmm['utc']]  # quadratic expansion of the data

        # computate of log-likelihoods for each frame and all Gaussian components
        gamma = -0.5 * data_sqr.dot(gmm['invCovs'].T) + data.dot(gmm['invCovMeans'].T) + gmm['gconsts']
        llh = GMM.log_sum_exp(gamma, axis=1)

        if return_accums == 0:
            return llh

        gamma = scipy.exp(gamma.T - llh)
        n = gamma.sum(axis=1)
        f = gamma.dot(data)

        if return_accums == 1:
            return llh, n, f

        s = gamma.dot(data_sqr)
        return llh, n, f, s

    @staticmethod
    def log_sum_exp(x, axis=0):
        """ Return natural logarithm of sum of exponents.

        Args:
            x (np.array): input
            axis (int):

        Returns:
            np.array: natural logarithm of sum of exponents
        """
        xmax = x.max(axis)
        ex = scipy.exp(x - np.expand_dims(xmax, axis))
        out = xmax + scipy.log(scipy.sum(ex, axis))
        not_finite = ~np.isfinite(xmax)
        out[not_finite] = xmax[not_finite]
        return out
