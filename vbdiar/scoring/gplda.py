import os
import operator
from functools import reduce

import numpy as np
from numpy.linalg import inv, slogdet


class GPLDA(object):
    """ Gaussian PLDA model.


    """
    cw = None
    cb = None
    mean = None
    p = None
    q = None
    k = None
    r = None
    s = None
    t = None
    u = None
    ct = None
    count = None
    initialized = False

    def __init__(self, path):
        """ Init all parameters needed for Gaussian PLDA Model score computation.

        Args:
            path(str): path with Gaussian PLDA model
        """
        self.cw = np.load(os.path.join(path, 'CW.npy'))
        self.cb = np.load(os.path.join(path, 'CB.npy'))
        self.mean = np.load(os.path.join(path, 'mu.npy'))

        self.initialize()

    def initialize(self):
        """ Initialize members for faster scoring. """
        self.ct = self.cw + self.cb
        self.p = inv(self.ct * 0.5) - inv(0.5 * self.cw + self.cb)
        self.q = inv(2 * self.cw) - inv(2 * self.ct)
        k1 = reduce(operator.mul, slogdet(0.5 * self.ct))
        k2 = reduce(operator.mul, slogdet(0.5 * self.cw + self.cb))
        k3 = reduce(operator.mul, slogdet(2 * self.ct))
        k4 = reduce(operator.mul, slogdet(2 * self.cw))
        self.k = 0.5 * (k1 - k2 + k3 - k4)
        self.r = 0.5 * (0.25 * self.p - self.q)
        self.s = 0.5 * (0.25 * self.p + self.q)
        self.t = 0.25 * np.dot(self.p, self.mean.T)
        u1 = 2 * np.dot(self.mean, 0.25 * self.p)
        self.u = self.k + np.dot(u1, self.mean.T)
        self.initialized = True

    def score(self, np_vec_1, np_vec_2):
        """ Compare two vectors using plda scoring metric. Function is symmetric.

        Args:
            np_vec_1 (np.array): array of vectors (e.g. nx250), depends on model
            np_vec_2 (np.array): array of vectors (e.g. nx250), depends on model

        Returns:
            2-dimensional np.array: scores matrix
        """
        if not self.initialized:
            raise ValueError('Model is not trained nor initialized.')
        np_vec_1 = np_vec_1.T.copy()
        np_vec_2 = np_vec_2.T.copy()
        mat1 = np.dot(self.r, np_vec_1) * np_vec_1
        mat2 = np.dot(self.r, np_vec_2) * np_vec_2
        vct1 = np.sum(mat1, axis=0, keepdims=True)
        vct2 = np.sum(mat2, axis=0, keepdims=True)
        vct3 = 2 * np.dot(self.t.T, np_vec_1)
        vct4 = 2 * np.dot(self.t.T, np_vec_2)
        mat3 = np.dot(np_vec_1.T, self.s)
        scores_matrix = 2 * np.dot(mat3, np_vec_2) + vct1.T + vct2 - vct3.T - vct4 + self.u
        return scores_matrix
