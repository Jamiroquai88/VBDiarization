#!/usr/bin/env python

import os
import numpy as np

from runplda import warp2us


class PLDA(object):
    """ Probabilistic linear discriminant analysis model.

    """
    def __init__(self, model_dir):
        """ Class constructor.

            :param model_dir: input directory with models.
            :type model_dir: str
        """
        self.model_dir = model_dir
        self.lda = np.load(os.path.join(model_dir, 'backend.LDA.npy'))
        self.mu_train = np.load(os.path.join(model_dir, 'backend.mu_train.npy'))
        self.plda_c = np.load(os.path.join(model_dir, 'backend.PLDA.c.npy'))
        self.plda_gamma = np.load(os.path.join(model_dir, 'backend.PLDA.Gamma.npy'))
        self.plda_k = np.load(os.path.join(model_dir, 'backend.PLDA.k.npy'))
        self.plda_lambda = np.load(os.path.join(model_dir, 'backend.PLDA.Lambda.npy'))

    def score(self, test, enroll, scale=1.0, shift=0.0):
        """ Score test and enroll against each other.

            :param test: test i-vectors
            :type test: numpy.array
            :param enroll: enroll i-vectors
            :type enroll: numpy.array
            :param scale: score scale
            :type scale: float
            :param shift: score shift
            :type shift: float
            :returns: PLDA scores
            :rtype: numpy.array
        """
        enroll = warp2us(enroll, self.lda, self.mu_train)
        test = warp2us(test, self.lda, self.mu_train)
        out = np.dot(enroll.dot(self.plda_lambda), test.T)
        out += (np.sum(enroll.dot(self.plda_gamma) * enroll, 1) + enroll.dot(self.plda_c))[:, np.newaxis]
        out += (np.sum(test.dot(self.plda_gamma) * test, 1) + test.dot(self.plda_c))[np.newaxis, :] + self.plda_k
        if scale is not None and shift is not None:
            return out * scale + shift
        else:
            return out
