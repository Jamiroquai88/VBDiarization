#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import numpy as np

from vbdiar.ivectors.gmm import GMM
from vbdiar.ivectors.extractor import Extractor


class Fea2Ivec(object):
    """ Class transforming features to i-vector.

    """

    def __init__(self, gmm_model, extractor_model):
        """ Initialize fe2ivec.

        Args:
            gmm_model (str): path to UBM GMM model
            extractor_model (str): path to T matrix
        """
        self.gmm = GMM(gmm_model)
        self.extractor = Extractor(extractor_model, self.gmm.num_g)

    def get_ivec(self, fea):
        """

        Args:
            fea (np.array): input features

        Returns:
            np.array: extracted i-vector
        """
        n_data, d_data = fea.shape
        seq_data = Extractor.split_seq(range(n_data), 4000)
        ww = 0
        lc = 0
        n = np.zeros(self.gmm.num_g, dtype=np.float32)
        ff = np.zeros((self.gmm.num_g, self.gmm.dim_f), dtype=np.float32)

        for ii in range(len(seq_data)):
            # TODO find out which variable does what
            dd = fea[seq_data[ii], :]
            l1, n1, f1 = GMM.gmm_eval(dd, self.gmm.ubm_gmm, return_accums=1)
            ww = ww + l1.sum()
            lc = lc + l1.shape[0]
            n = n + n1
            ff = ff + f1

        n, ff = self.gmm.normalize_stats(n, ff)

        ff = Extractor.row(ff.astype(self.extractor.v_matrix.dtype))
        n = Extractor.row(n.astype(self.extractor.v_matrix.dtype))
        return Extractor.estimate_ivec(n, ff, self.extractor.v_matrix, self.extractor.mvvt).T
