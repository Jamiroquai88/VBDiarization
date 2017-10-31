#!/usr/bin/env python

import h5py
import numpy as np


class PLDA(object):
    """ Probabilistic linear discriminant analysis model.

    """
    def __init__(self, model_path):
        """ Initialize PLDA model.

        Args:
            model_path (str): path to model in .h5 format
        """
        self.v, self.u, self.d, self.mu = PLDA.load(model_path)

    @staticmethod
    def load(model_path):
        """ Load PLDA model parts.

        Args:
            model_path (str): path to model

        Returns:
            tuple: V, U, D, mu
        """
        model = h5py.File(model_path, 'r')
        return model['V'][:], model['U'][:], model['D'][:], model['mu'][:]

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
            :returns: PLDA scoress
            :rtype: numpy.array
        """
        enroll = PLDA.warp2us(enroll, self.lda, self.mu)
        test = PLDA.warp2us(test, self.lda, self.mu)
        out = np.dot(enroll.dot(self.plambda), test.T)
        out += (np.sum(enroll.dot(self.gamma) * enroll, 1) + enroll.dot(self.c))[:, np.newaxis]
        out += (np.sum(test.dot(self.gamma) * test, 1) + test.dot(self.c))[np.newaxis, :] + self.k
        if scale is not None and shift is not None:
            return out * scale + shift
        else:
            return out

