#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import ctypes

import h5py
import numpy as np

MKL = ctypes.cdll.LoadLibrary('libmkl_rt.so')


class Extractor(object):
    """ i-vector extractor """

    def __init__(self, model_path, n_gauss):
        """ Initialize model.

        Args:
            model_path (str): path to model (T matrix)
            n_gauss (int): number of Gaussian mixture components
        """
        self.v_matrix = Extractor.load(model_path)
        self.mvvt = Extractor.compute_vtv(self.v_matrix, n_gauss)

    @staticmethod
    def load(model_path):
        """

        Args:
            model_path (str): path to T matrix

        Returns:
            np.array: loaded model
        """
        return h5py.File(model_path, 'r')['v'][:]

    @staticmethod
    def row(v):
        """ Reshape to row vector.

        Args:
            v (np.array): input matrix

        Returns:
            np.array: reshaped row vector
        """
        return v.reshape((1, v.size))

    @staticmethod
    def split_seq(seq, size):
        """  Split up sequence to pieces.

        Args:
            seq (np.array):
            size (int):

        Returns:
            list: splitted sequences
        """
        return [seq[ii:ii + size] for ii in range(0, len(seq), size)]

    @staticmethod
    def compute_vtv(v_matrix, n_gauss):
        """ Compute vtv - 3d matrix using MKL implementation.

        Args:
            v_matrix (np.array): input v matrix
            n_gauss (int): number of gaussian components

        Returns:
            np.array: vtv matrix
        """
        v_dim = v_matrix.shape[1]  # subspace dim
        f_dim = v_matrix.shape[0] / n_gauss  # feature dim
        r, c = Extractor.get_rfpf_shape(v_dim)

        # Allocate space if necessary
        out = np.zeros((r, c, n_gauss), dtype=v_matrix.dtype, order='F')

        # reshape v to convenience
        v3d = v_matrix.reshape((n_gauss, f_dim, v_dim))

        for ii in range(n_gauss):
            out[:, :, ii] = Extractor.rank_k_update(v3d[ii, :, :].T, out=out[:, :, ii])

        return out

    @staticmethod
    def get_rfpf_shape(n, transr='N'):
        """ Get shape for RFPF format.

        Args:
            n (int): input dimension
            transr (str): transpose matrix

        Returns:
            tuple: rfpf shape
        """
        if n % 2 == 0:
            out_rows = n + 1
            out_cols = int(n / 2)
        else:
            out_rows = n
            out_cols = int((n + 1) / 2)
        if transr != 'T':
            out_shape = (out_rows, out_cols)
        else:
            out_shape = (out_cols, out_rows)
        return out_shape

    @staticmethod
    def rank_k_update(a_matrix, trans='N', transr='N', uplo='U', out=None):
        # TODO docstring, find out what it does
        """

        Args:
            a_matrix:
            trans:
            transr (str): transpose matrix
            uplo:
            out:

        Returns:

        """
        a_cols, a_rows = a_matrix.shape

        if a_matrix.flags.c_contiguous:
            a_matrix = np.asfortranarray(a_matrix)
            # TODO raise ValueError('Input matrix A needs to be F-contiguous.')

        lda = a_matrix.strides[1] / a_matrix.dtype.itemsize

        if trans == 'N':
            n = a_cols
            k = a_rows
        else:
            n = a_rows
            k = a_cols

        out_shape = Extractor.get_rfpf_shape(n, transr)

        if out is None:
            out = np.empty(out_shape, a_matrix.dtype, order='F')

        else:
            if out.shape != out_shape:
                raise ValueError('Out-array is badly defined')

            if out.dtype != a_matrix.dtype:
                raise ValueError('Weird conversion')

        if a_matrix.dtype == np.float64:
            bl_transr = ctypes.byref(ctypes.c_char(transr))
            bl_uplo = ctypes.byref(ctypes.c_char(uplo))
            bl_trans = ctypes.byref(ctypes.c_char(trans))
            bl_n = ctypes.byref(ctypes.c_int(n))
            bl_k = ctypes.byref(ctypes.c_int(k))
            bl_alpha = ctypes.byref(ctypes.c_double(1.0))
            bl_a = ctypes.c_void_p(a_matrix.ctypes.data)
            bl_lda = ctypes.byref(ctypes.c_int(lda))
            bl_beta = ctypes.byref(ctypes.c_double(0.0))
            bl_c = ctypes.c_void_p(out.ctypes.data)

            MKL.dsfrk(bl_transr, bl_uplo, bl_trans, bl_n, bl_k, bl_alpha, bl_a, bl_lda, bl_beta, bl_c)

        elif a_matrix.dtype == np.float32:
            bl_transr = ctypes.byref(ctypes.c_char(transr))
            bl_uplo = ctypes.byref(ctypes.c_char(uplo))
            bl_trans = ctypes.byref(ctypes.c_char(trans))
            bl_n = ctypes.byref(ctypes.c_int(n))
            bl_k = ctypes.byref(ctypes.c_int(k))
            bl_alpha = ctypes.byref(ctypes.c_float(1.0))
            bl_a = ctypes.c_void_p(a_matrix.ctypes.data)
            bl_lda = ctypes.byref(ctypes.c_int(lda))
            bl_beta = ctypes.byref(ctypes.c_float(0.0))
            bl_c = ctypes.c_void_p(out.ctypes.data)

            MKL.ssfrk(bl_transr, bl_uplo, bl_trans, bl_n, bl_k, bl_alpha, bl_a, bl_lda, bl_beta, bl_c)

        return out

    @staticmethod
    def to_rfpf(a, transr='N', uplo='U', out=None):
        """ Converts the original matrix to RFPF based on the Transr and Uplo
            parameters. Note that Fortran of A is required, otherwise, the matrix
            will be taken as a transpose of the original one.

            Args:
                a (np.array): original matrix
                transr (str): transpose matrix
                uplo (str):
                out (np.array): output matrix

            Returns:
                np.array: matrix in RFPF format

        """
        a_cols, a_rows = a.shape

        if a_cols != a_rows:
            raise ValueError('Matrix is not square.')

        if a.flags.c_contiguous:
            a = np.asfortranarray(a)
            # TODO raise ValueError('Input matrix A needs to be F-contiguous.')

        lda = a.strides[1] / a.dtype.itemsize

        if out is None:
            out_shape = Extractor.get_rfpf_shape(a_cols, transr)
            out = np.empty(out_shape, a.dtype, order='F')

        ii = ctypes.c_int()

        bl_transr = ctypes.byref(ctypes.c_char(transr))
        bl_uplo = ctypes.byref(ctypes.c_char(uplo))
        bl_n = ctypes.byref(ctypes.c_int(a_cols))
        bl_a = ctypes.c_void_p(a.ctypes.data)
        bl_lda = ctypes.byref(ctypes.c_int(lda))
        bl_arf = ctypes.c_void_p(out.ctypes.data)
        bl_info = ctypes.byref(ii)

        if a.dtype == np.float64:
            MKL.dtrttf(bl_transr, bl_uplo, bl_n, bl_a, bl_lda, bl_arf, bl_info)

        elif a.dtype == np.float32:
            MKL.strttf(bl_transr, bl_uplo, bl_n, bl_a, bl_lda, bl_arf, bl_info)

        if ii < 0:
            raise ValueError('Argument {} of dtrttf is bad'.format(ii))

        return out

    @staticmethod
    def solve(aorig, borig, uplo='U'):
        if borig.flags.c_contiguous:
            borig = np.asfortranarray(borig)
            # TODO raise ValueError('Input matrix B needs to be F-contiguous.')

        if borig.dtype != aorig.dtype:
            raise ValueError('Matrices are of different types.')

        b = borig.copy(order='F')
        a = aorig.copy(order='F')

        if a.shape[0] > a.shape[1]:
            transr = 'N'
        else:
            transr = 'T'

        nrows, nrhs = b.shape

        ldb = int(b.strides[1] / b.dtype.itemsize) if b.dtype.itemsize else 0

        nt = nrows * (nrows + 1) / 2

        if nt != a.size:
            raise ValueError('Matrices are not aligned.')

        ii = ctypes.c_int(0)

        bl_transr = ctypes.byref(ctypes.c_char(transr))
        bl_uplo = ctypes.byref(ctypes.c_char(uplo))
        bl_n = ctypes.byref(ctypes.c_int(nrows))
        bl_nrhs = ctypes.byref(ctypes.c_int(nrhs))
        bl_a = ctypes.c_void_p(a.ctypes.data)
        bl_ldb = ctypes.byref(ctypes.c_int(ldb))
        bl_b = ctypes.c_void_p(b.ctypes.data)
        bl_info = ctypes.byref(ii)

        if a.dtype == np.float64:
            MKL.dpftrf(bl_transr, bl_uplo, bl_n, bl_a, bl_info)
            MKL.dpftrs(bl_transr, bl_uplo, bl_n, bl_nrhs, bl_a, bl_b, bl_ldb, bl_info)

        elif a.dtype == np.float32:
            MKL.spftrf(bl_transr, bl_uplo, bl_n, bl_a, bl_info)
            MKL.spftrs(bl_transr, bl_uplo, bl_n, bl_nrhs, bl_a, bl_b, bl_ldb, bl_info)

        return b

    @staticmethod
    def estimate_ivec(nt, ft, v_matrix, vtv_matrix, eye=None):
        """ Estimate i-vector based on statistics.

        Args:
            nt:
            ft:
            v_matrix (np.array): input v matrix
            vtv_matrix (np.array): input precomputed vtv matrix
            eye:

        Returns:
            np.array: estimated i-vector
        """
        v_dim = v_matrix.shape[1]
        n_gauss = nt.shape[1]

        # Construct eye if necessary
        if eye is None:
            eye = Extractor.to_rfpf(np.eye(v_dim, dtype=v_matrix.dtype).T)

        it = eye.T.reshape((1, -1))
        vtvt = vtv_matrix.T.reshape((n_gauss, -1))

        b = np.dot(ft, v_matrix).T
        lt = np.dot(nt, vtvt) + it

        l = lt.reshape((vtv_matrix.shape[1], vtv_matrix.shape[0])).T

        out = Extractor.solve(l, b)

        return out
