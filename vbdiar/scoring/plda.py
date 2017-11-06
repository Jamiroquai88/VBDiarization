#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import h5py
import numpy as np
from scipy.io.idl import AttrDict
from scipy.sparse import coo_matrix

from vbdiar.utils.utils import Utils


class PLDA(object):

    V = None
    U = None
    D = None
    mc = None
    mu = None
    _R = None
    _T = None
    _y = None
    _x = None
    Syy = None
    vdim = None
    srank = None
    crank = None
    noise_type = None

    def init(self, vdim, srank, crank, diag="isotropic"):
        """

        Args:
            vdim:
            srank:
            crank:
            diag:
        """
        self.vdim = vdim
        self.srank = srank
        self.crank = crank
        self.noise_type = diag

        np.random.seed(7)
        self.V = np.array(np.random.randn(vdim, srank), np.float) * 1e0
        self.U = np.array(np.random.randn(vdim, crank), np.float) * 1e0
        if diag == "zero":
            self.D = np.zeros((vdim, 1), np.float)
        else:
            self.D = np.ones((vdim, 1), np.float)

        self.mu = np.zeros((vdim, 1), np.float)

        # expectations
        self._R = np.eye(self.srank + self.crank, self.srank + self.crank)
        self._T = np.zeros((self.srank + self.crank, self.vdim))
        self.Syy = np.zeros((self.srank, self.srank))
        # point estimates
        self._y = None
        self._x = None

    def __str__(self):
        return "= PLDA model d=%d,s=%d,c=%d,type=%s" % (self.vdim, self.srank, self.crank, self.noise_type)

    def __init__(self, model_file):
        """

        Args:
            model_file:

        Returns:

        """
        model = h5py.File(model_file, 'r')
        v, u, d, noise_type, mu = model['V'][:], model['U'][:], model['D'][:], model['noise_type'], model['mu'][:]
        vdim, srank = v.shape
        crank = u.shape[1]
        self.init(vdim, srank, crank, noise_type)
        self.V = v
        self.U = u
        self.D = d
        self.mu = mu
        self.cache_statistics()

    @staticmethod
    def symmetrize(a):
        a = a + a.T
        a *= 0.5
        return a

    @staticmethod
    def logdet_chol(chol_m):
        """Return the log determinant of a matrix given the cholesky decomposition"""
        return 2 * np.sum(np.log(np.diagonal(chol_m)))

    @staticmethod
    def invhandle(m, func_only=True):
        """Return a function handle on multiplying a matrix
           by the inverse of the marix provided in this function
           if func_only=False then return: [logdet,chol,handle]
           (Niko's style)
           This is used when scoring the PLDA system
           """
        cm = np.linalg.cholesky(m)

        def h(a):
            return np.linalg.solve(cm.T, np.linalg.solve(cm, a.T))

        if func_only:
            return h
        else:
            logdet = PLDA.logdet_chol(cm)
            return [logdet, cm, h]

    def cache_statistics(self):
        """
            various values independant of the data in the cache object
        """
        # values independant of the data
        mc = AttrDict()
        mc.UD = self.U*self.D
        mc.VD = self.V*self.D
        mc.J = np.dot(self.U.T, mc.VD)

        mc.K = np.sqrt(self.D) * self.U
        mc.K = PLDA.symmetrize(np.dot(mc.K.T, mc.K)+np.eye(self.crank))

        mc.iK = np.linalg.inv(mc.K)
        mc.P0 = mc.VD.T-np.dot(np.dot(mc.J.T, mc.iK), mc.UD.T)
        self.mc = mc
        return mc

    def score_with_constant_n(self, nt, ft, nt2, ft2):
        """  Compute verification score
             This changes for every pair nT nt and can be done only once for 1 session training
             fT = P*y_head = (V'D-J'iK*U'D)*f, where f is the first order stats
        """
        # Covariance and first order stats preparation
        Py = PLDA.symmetrize(np.dot(self.mc.P0, self.V))  # (DV-iKJDU)V=VDV-JiKJ

        # P in eq. 31, EM for PLDA by Niko
        Py_Train = nt * Py + np.eye(self.srank)
        Py_Test = nt2 * Py + np.eye(self.srank)
        Py_SameSpk = (nt2 + nt) * Py + np.eye(self.srank)

        [logdetQ1, cholQ1, hQ1] = PLDA.invhandle(Py_Train, func_only=False)
        [logdetQ2, cholQ2, hQ2] = PLDA.invhandle(Py_Test, func_only=False)
        [logdetQ12, cholQ12, hQ12] = PLDA.invhandle(Py_SameSpk, func_only=False)
        # fT*hQ1(fT.T) is terms y_head*P*y_head in eq 30, EM for PLDA by Niko
        Q1 = 0.5 * (logdetQ1 + sum(ft * hQ12(ft.T)) - sum(ft * hQ1(ft.T)))
        Q2 = 0.5 * (logdetQ2 + sum(ft2 * hQ12(ft2.T)) - sum(ft2 * hQ2(ft2.T)))
        # Q2=Q2[np.newaxis,:] (can be dangerous)
        Q1 = Q1[:, np.newaxis]
        scores = np.dot(ft.T, hQ12(ft2.T))
        Q2 -= 0.5 * logdetQ12
        scores += Q1
        scores += Q2
        return scores

    def prepare_stats(self, data):
        """ Convert data (e.g. i-vectors) to statistics useful for scoring
            Input:
                data:      ivect-dim x nb_ex matrix
                seg2model: defines of multisession enrollment (or test). It is
                           2 column array of integer indices mapping rows of data
                           (1st column) to rows of output statistics (2nd column).
                           By default (seg2model=None) each vector in data has its
                           own output vector of statistics (i.e. single session
                           enrollment or test). seg2model can be also represented
                           by coo_matrix or by 1D array of labels if each input
                           vector belongs to exactly one enrollment.
            Output: a AttrDict object with
                    N: vector of counts for each speaker
                    F: array of sum of ivector for each speaker transformed by P0
                       to srank dimensionality
        """
        seg2model = np.arange(len(data))
        if seg2model.ndim == 1:
            seg2model = np.c_[np.arange(len(data)), seg2model]
        seg2model = coo_matrix((np.ones(len(seg2model)), (seg2model[:, 0], seg2model[:, 1])))

        stats = AttrDict()
        stats.N = np.array(seg2model.T.sum(1))
        stats.F = np.dot(self.mc.P0, seg2model.T.dot(data - self.mu).T).T
        return stats

    def score(self, test, enroll):
        """
            Score ivectors based on the PLDA model
            Input:
                PLDA object. PLDA.V and PLDA.U gives the subspaces
                stats objects: enroll and test
            Output:
                2D array of scores of all possibilities
        """
        test = test - self.mu
        enroll = enroll - self.mu
        test = Utils.l2_norm(test)
        enroll = Utils.l2_norm(enroll)
        Tstats = self.prepare_stats(test)
        tstats = self.prepare_stats(enroll)
        # Create scores
        scores = np.zeros((len(Tstats.N), len(tstats.N)), 'f')
        (a, b) = scores.shape

        # Score for each uniq combination of N enroll and M test sessions (only enroll for now)
        for n_enroll_sessions in np.unique(Tstats.N):
            idxs_T = np.where(Tstats.N == n_enroll_sessions)[0]
            for n_test_sessions in np.unique(tstats.N):
                idxs_t = np.where(tstats.N == n_test_sessions)[0]
                scores[np.ix_(idxs_T, idxs_t)] = self.score_with_constant_n(n_enroll_sessions, Tstats.F[idxs_T, :].T,
                                                                            n_test_sessions, tstats.F[idxs_t, :].T)
        return scores.T


