#!/usr/bin/env python

import os
import re
import pickle
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans

from plda import PLDA
from tools import Tools
from kmeans import KMeans
from cluster import Cluster
from normalization import Normalization
from tools import loginfo, logwarning
from user_exception import DiarizationException


class Diarization(Normalization):

    def __init__(self, input_list, norm_list, ivecs_dir, out_dir, plda_model_dir):
        super(Diarization, self).__init__(ivecs_dir, norm_list, plda_model_dir)
        self.input_list = input_list
        self.ivecs_dir = ivecs_dir
        self.out_dir = out_dir
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
                loginfo('[Diarization.load_ivecs] Loading pickle file {} ...'.format(line.rstrip().split()[0]))
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
                            # mean = np.mean(ivec_set.get_all())
                            # for ivec in ivec_set:
                            #     ivec.data = ivec.data - mean
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
                    num_speakers = ivecset.num_speakers
                    sklearnkmeans = sklearnKMeans(n_clusters=num_speakers).fit(ivecs)
                    centroids = self.plda_kmeans(ivecs, sklearnkmeans.cluster_centers_, num_speakers)
                else:
                    num_speakers, centroids = self.get_num_speakers(ivecs)
                if self.norm_list is None:
                    scores_dict[name] = self.plda.score(ivecs, centroids)
                else:
                    scores_dict[name] = self.s_norm(ivecs, centroids)
            else:
                logwarning('[Diarization.score] No i-vectors to score in {}.'.format(ivecset.name))
        return scores_dict

    def dump_rttm(self, scores):
        for ivecset in self.ivecs:
            if ivecset.size() > 0:
                name = ivecset.name
                # dirty trick, will be removed, watch out
                if 'beamformed' in ivecset.name:
                    ivecset.name = re.sub('beamformed/', '', ivecset.name)
                # # # # # # # # # # # # # # # # # # # # #
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

    def get_num_speakers(self, ivecs, min_spk=2, max_spk=10):
        print ivecs.shape
        avg, centroids_list = [], []
        for ii in range(min_spk, max_spk):
            sklearnkmeans = sklearnKMeans(n_clusters=ii).fit(ivecs)
            centroids = self.plda_kmeans(ivecs, sklearnkmeans.cluster_centers_, ii)
            centroids_list.append(centroids)
            scores = self.s_norm(centroids, centroids)
            print 'Speakers', ii
            print 'Avg', np.sum(np.tril(scores, -1)) / (ii * (ii + 1) / 2)
            avg.append(np.sum(np.tril(scores, -1)) / (ii * (ii + 1) / 2))
        print 'Min:', avg.index(min(avg)) + min_spk, min(avg)
        return avg.index(min(avg)) + min_spk, centroids_list[avg.index(min(avg)) - min_spk]


# c = Cluster()
        # if ivecset.size() > 0:
        #     scores = self.s_norm(ivecset.get_all(), ivecset.get_all())
        #     shape = scores.shape[0]
        #     for ii in range(shape):
        #         c.add(ii, 0)
        #     for ii in range(shape):
        #         for jj in range(ii + 1, shape):
        #             if scores[ii][jj] > thr:
        #                     c.merge(ii, jj)
        #     print 'Size:', c.size()
        #     return c.size()
        # else:
        #     logwarning('[Diarization.score] No i-vectors to score in {}.'.format(ivecset.name))

