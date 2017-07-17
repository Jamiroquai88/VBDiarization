#!/usr/bin/env python

import os
import re
import pickle
import numpy as np
from pyannote.core import Segment
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from pyannote.core import Annotation
from sklearn.preprocessing import normalize
from pyannote.metrics.diarization import DiarizationErrorRate

from plda import PLDA
from tools import Tools
from ivec import IvecSet
from tools import loginfo, logwarning
from user_exception import DiarizationException


class Diarization(object):

    def __init__(self, input_list, ivecs_dir, out_dir, plda_model_dir):
        self.input_list = input_list
        self.ivecs_dir = ivecs_dir
        self.out_dir = out_dir
        self.plda = PLDA(plda_model_dir)
        self.ivecs = list(self.load_ivecs())

    def get_ivec(self, name):
        for ii in self.ivecs:
            if name == ii.name:
                return ii
        raise DiarizationException(
            '[Diarization.get_ivec] Name of the set not found.'
        )

    def load_ivecs(self):
        with open(self.input_list, 'r') as f:
            for line in f:
                loginfo('[Diarization.load_ivecs] Loading pickle file for {} ...'.format(line.rstrip().split()[0]))
                line = line.rstrip()
                try:
                    if len(line.split()) == 1:
                        with open(os.path.join(self.ivecs_dir, line + '.pkl')) as i:
                            yield pickle.load(i)
                    elif len(line.split()) == 2:
                        file_name = line.split()[0]
                        num_spks = int(line.split()[1])
                        with open(os.path.join(self.ivecs_dir, file_name + '.pkl')) as i:
                            ivec_set = pickle.load(i)
                            ivec_set.num_speakers = num_spks
                            yield ivec_set
                    else:
                        raise DiarizationException(
                            '[Diarization.load_ivecs] Unexpected number of columns in input list {}.'.format(
                                self.input_list)
                        )
                except IOError:
                    logwarning(
                        '[Diarization.load_ivecs] No pickle file found for {}.'.format(line.rstrip().split()[0]))

    def score(self):
        scores_dict = {}
        for ivecset in self.ivecs:
            name = os.path.normpath(ivecset.name)
            ivecs = ivecset.get_all()
            loginfo('[Diarization.score] Scoring {} ...'.format(name))
            size = ivecset.size()
            if size > 0:
                if ivecset.num_speakers is not None:
                    kmeans = KMeans(n_clusters=ivecset.num_speakers).fit(ivecs)
                    scores_dict[name] = self.plda.score(ivecs, kmeans.cluster_centers_)
                else:
                    pass
            else:
                logwarning('[Diarization.score] No i-vectors to score in {}.'.format(ivecset.name))
        return scores_dict

    def dump_rttm(self, scores):
        for ivecset in self.ivecs:
            if ivecset.size() > 0:
                name = ivecset.name
                reg_name = re.sub('/.*', '', ivecset.name)
                Tools.mkdir_p(os.path.join(self.out_dir, os.path.dirname(name)))
                with open(os.path.join(self.out_dir, name + '.rttm'), 'w') as f:
                    for i, ivec in enumerate(ivecset.ivecs):
                        start, end = ivec.window_start, ivec.window_end
                        idx = np.argmax(scores[name].T[i])
                        f.write('SPEAKER {} 1 {} {} <NA> <NA> {}_spkr_{} <NA>\n'.format(
                            reg_name, float(start / 1000.0), float((end - start) / 1000.0), reg_name, idx))
            else:
                logwarning('[Diarization.score] No i-vectors to dump in {}.'.format(ivecset.name))

# % load_ext autoreload
# % autoreload 2
# from lib.diarization import Diarization
# diar = Diarization('../db/AMI/lists/AMI_Array1-01_dev-eval.lst', '../db/AMI/i-vectors/', '../db/AMI/results', '../models/')


