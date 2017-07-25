#!/usr/bin/env python

import os
import numpy as np

from plda import PLDA
from kmeans import KMeans
from tools import loginfo, logwarning


class Normalization(object):

    def __init__(self, ivecs_dir, norm_list, plda_model_dir):
        self.ivecs_dir = ivecs_dir
        self.norm_list = norm_list
        self.plda = PLDA(plda_model_dir)
        if self.norm_list is not None:
            self.norm_ivecs = np.array(list(self.load_norm_ivecs()))

    def load_norm_ivecs(self):
        with open(self.norm_list, 'r') as f:
            for line in f:
                line = line.rstrip()
                loginfo('[Diarization.load_norm_ivecs] Loading npy file {} ...'.format(line))
                try:
                    yield np.load('{}.npy'.format(os.path.join(self.ivecs_dir, line))).flatten()
                except IOError:
                    logwarning('[Diarization.load_norm_ivecs] No pickle file found for {}.'.format(line))

    def s_norm(self, test, enroll):
        a = self.plda.score(test, self.norm_ivecs)
        b = self.plda.score(enroll, self.norm_ivecs)
        c = self.plda.score(enroll, test)
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
