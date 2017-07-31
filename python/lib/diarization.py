#!/usr/bin/env python

import os
import re
import pickle
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from pyannote.core import Annotation, Segment
from sklearn.cluster import KMeans as sklearnKMeans
from pyannote.metrics.diarization import DiarizationErrorRate

from tools import Tools
from kmeans import KMeans
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
                    centroids = KMeans(sklearnkmeans.cluster_centers_, num_speakers, self.plda).fit(ivecs)
                else:
                    num_speakers, centroids = self.get_num_speakers(ivecs)
                if self.norm_list is None:
                    scores_dict[name] = self.plda.score(ivecs, centroids, self.scale, self.shift)
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
                logwarning('[Diarization.dump_rttm] No i-vectors to dump in {}.'.format(ivecset.name))

    def get_der(self, ref_file, scores):
        ref, hyp = self.init_annotations()
        with open(ref_file, 'r') as f:
            for line in f:
                _, name, _, start, duration, _, _, speaker, _ = line.split()
                ref[name][Segment(float(start), float(start) + float(duration))] = speaker
        for ivecset in self.ivecs:
            if ivecset.size() > 0:
                name, reg_name = ivecset.name, ivecset.name
                # dirty trick, will be removed, watch out
                if 'beamformed' in name:
                    reg_name = re.sub('beamformed/', '', name)
                # # # # # # # # # # # # # # # # # # # # #
                reg_name = re.sub('/.*', '', reg_name)
                for i, ivec in enumerate(ivecset.ivecs):
                    start, end = ivec.window_start / 1000.0, ivec.window_end / 1000.0
                    hyp[reg_name][Segment(start, end)] = np.argmax(scores[name].T[i])
            else:
                logwarning('[Diarization.get_der] No i-vectors to dump in {}.'.format(ivecset.name))
        der = DiarizationErrorRate()
        der.collar = 0.25
        names, values, summ = [], [], 0.0
        for name in ref.keys():
            names.append(name)
            der_num = der(ref[name], hyp[name]) * 100
            values.append(der_num)
            summ += der_num
            loginfo('[Diarization.get_der] {} DER = {}'.format(name, '{0:.3f}'.format(der_num)))
        loginfo('[Diarization.get_der] Average DER = {}'.format('{0:.3f}'.format(summ / float(len(ref.keys())))))
        Diarization.plot_der(names, values)

    def init_annotations(self):
        ref, hyp = {}, {}
        for ivecset in self.ivecs:
            if ivecset.size() > 0:
                name = ivecset.name
                # dirty trick, will be removed, watch out
                if 'beamformed' in name:
                    name = re.sub('beamformed/', '', name)
                # # # # # # # # # # # # # # # # # # # # #
                name = re.sub('/.*', '', name)
                ref[name], hyp[name] = Annotation(), Annotation()
        return ref, hyp

    def get_num_speakers(self, ivecs, min_speakers=2, max_speakers=6):
        avg, centroids_list = [], []
        features = []
        for num_speakers in range(min_speakers, max_speakers + 1):
            sklearnkmeans = sklearnKMeans(n_clusters=num_speakers).fit(ivecs)
            centroids = KMeans(sklearnkmeans.cluster_centers_, num_speakers, self.plda).fit(ivecs)
            centroids_list.append(centroids)
            scores = self.s_norm(centroids, centroids)[np.tril_indices(num_speakers, -1)]
            features.append(Normalization.get_features(scores))
        num_speakers = np.argmax(np.sum(self.model.test(features, prob=True), axis=0))
        # raw_input('ENTER')
        return num_speakers + min_speakers, centroids_list[num_speakers]

    @staticmethod
    def plot_der(names, values):
        values = [x for (y, x) in sorted(zip(names, values))]
        names = sorted(names)
        print values, names
        trace1 = go.Scatter(x=names, y=values)
        layout = dict(title='Diarization Error Rate - Beamformed',
                      xaxis=dict(title='Recording'),
                      yaxis=dict(title='DER [%]'),
                      )
        fig = dict(data=[trace1], layout=layout)
        py.iplot(fig, filename='Diarization Error Rate')
