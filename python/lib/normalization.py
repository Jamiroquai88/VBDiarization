#!/usr/bin/env python

import os
import pickle
import numpy as np

from plda import PLDA
from tools import loginfo, logwarning


class Normalization(object):
    """ Speaker normalization S-Norm. Handles also some other operation as calibration, detecting number of speakers.

    """
    def __init__(self, ivecs_dir, norm_list, plda_model_dir):
        """ Class constructor.

            :param ivecs_dir: path to directory with i-vectors
            :type ivecs_dir: str
            :param norm_list: path to list with files relative to directory with i-vectors
            :type norm_list: str
            :param plda_model_dir: path to directory with models
            :type plda_model_dir: str
        """
        self.ivecs_dir = ivecs_dir
        self.scale, self.shift, self.model = None, None, None
        self.norm_list = norm_list
        self.plda = PLDA(plda_model_dir)
        if self.norm_list is not None:
            self.norm_ivecs = np.array(list(self.load_norm_ivecs()))
        else:
            self.norm_ivecs = None

    def load_norm_ivecs(self):
        """ Load normalization i-vectors, scale and shift files and also pretrained model.

            :returns: i-vectors
            :rtype: numpy.array
        """
        line = None
        with open(self.norm_list, 'r') as f:
            for line in f:
                line = line.rstrip()
                loginfo('[Diarization.load_norm_ivecs] Loading npy file {} ...'.format(line))
                try:
                    yield np.load('{}.npy'.format(os.path.join(self.ivecs_dir, line))).flatten()
                except IOError:
                    logwarning('[Diarization.load_norm_ivecs] No pickle file found for {}.'.format(line))
        self.scale = np.load(os.path.join(self.ivecs_dir, os.path.dirname(line), 'scale.npy'))
        self.shift = np.load(os.path.join(self.ivecs_dir, os.path.dirname(line), 'shift.npy'))
        try:
            with open(os.path.join(self.ivecs_dir, os.path.dirname(line), 'model.pkl')) as f:
                self.model = pickle.load(f)
        except IOError:
            logwarning('[Diarization.load_norm_ivecs] No pretrained model found.')

    def s_norm(self, test, enroll):
        """ Run S-Norm on input i-vectors.

            :param test: test i-vectors
            :type test: numpy.array
            :param enroll: enroll i-vectors
            :type enroll: numpy.array
            :returns: scores matrix
            :rtype: numpy.array
        """
        a = self.plda.score(test, self.norm_ivecs, scale=self.scale, shift=self.shift)
        b = self.plda.score(enroll, self.norm_ivecs, scale=self.scale, shift=self.shift)
        c = self.plda.score(enroll, test, scale=self.scale, shift=self.shift)
        scores = []
        for ii in range(test.shape[0]):
            test_scores = []
            for jj in range(enroll.shape[0]):
                test_mean, test_std = np.mean(a.T[ii]), np.std(a.T[ii])
                enroll_mean, enroll_std = np.mean(b.T[jj]), np.std(b.T[jj])
                s = c[ii][jj]
                test_scores.append((((s - test_mean) / test_std + (s - enroll_mean) / enroll_std) / 2))
            scores.append(test_scores)
        return np.array(scores).T

    @staticmethod
    def get_features(scores):
        """ Compute features from input scores.

            :param scores: input scores
            :type scores: list
            :returns: mean, std and median
            :rtype: tuple
        """
        return np.mean(scores), np.std(scores), np.median(scores)
