#!/usr/bin/env python

import os
import re
import pickle

import numpy as np
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.clustering.pldakmeans import PLDAKMeans
from vbdiar.scoring.normalization import Normalization
from vbdiar.utils.utils import Utils, loginfo, logwarning
from vbdiar.utils.user_exception import DiarizationException


class Diarization(object):
    """ Diarization class used as main diarization focused implementation.

    """
    def __init__(self, input_list, ivecs, norm=None, plda=None):
        """ Initialize diarization class.

        Args:
            input_list (str): path to list of input files
            ivecs (str|list): path to directory containing i-vectors or list of IvecSet instances
            norm (Normalization): instance of class Normalization
            plda (PLDA): instance of class PLDA
        """
        self.input_list = input_list
        if isinstance(ivecs, str):
            self.ivecs_dir = ivecs
            self.ivecs = list(self.load_ivecs())
        else:
            self.ivecs = ivecs
        self.norm = norm
        self.plda = plda

    def get_ivec(self, name):
        """ Get i-ivector set by name.

            :param name: name of the set
            :type name: str
            :returns: set of i-vectors
            :rtype: IvecSet
        """
        for ii in self.ivecs:
            print ii.name
            if name == ii.name:
                return ii
        raise DiarizationException(
            'Name of the set not found - {}.'.format(name)
        )

    def load_ivecs(self):
        """ Load i-vectors stored as pickle files.

            :returns: list of i-vectors sets
            :rtype: list
        """
        with open(self.input_list, 'r') as f:
            for line in f:
                loginfo('Loading pickle file {} ...'.format(line.rstrip().split()[0]))
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
                            'Unexpected number of columns in input list {}.'.format(
                                self.input_list)
                        )
                except IOError:
                    logwarning(
                        'No pickle file found for {}.'.format(line.rstrip().split()[0]))

    def score_ivec(self, max_num_speakers):
        """ Score i-vectors agains speaker clusters.

            :returns: PLDA scores
            :rtype: numpy.array
        """
        scores_dict = {}
        for ivecset in self.ivecs:
            name = os.path.normpath(ivecset.name)
            ivecs = ivecset.get_all()
            loginfo('Scoring {} ...'.format(name))
            size = ivecset.size()
            if size > 0:
                if ivecset.num_speakers is not None:
                    num_speakers = ivecset.num_speakers
                    sklearnkmeans = sklearnKMeans(n_clusters=num_speakers).fit(ivecs)
                    if self.plda is None:
                        centroids = sklearnkmeans.cluster_centers_
                    else:
                        centroids = PLDAKMeans(sklearnkmeans.cluster_centers_, num_speakers, self.plda).fit(ivecs)
                else:
                    xm = xmeans(ivecs, kmax=max_num_speakers)
                    xm.process()
                    centroids = np.array(xm.get_clusters())
                if self.norm is None:
                    if self.plda is None:
                        ivecs = Utils.l2_norm(ivecs)
                        centroids = Utils.l2_norm(centroids)
                        scores_dict[name] = cosine_similarity(ivecs, centroids).T
                    else:
                        scores_dict[name] = self.plda.score(ivecs, centroids)
                else:
                    ivecs = Utils.l2_norm(ivecs)
                    centroids = Utils.l2_norm(centroids)
                    scores_dict[name] = self.norm.s_norm(ivecs, centroids)
            else:
                logwarning('No i-vectors to score in {}.'.format(ivecset.name))
        return scores_dict

    def dump_rttm(self, scores, out_dir):
        """

        Args:
            scores:
            out_dir:

        Returns:

        """
        for ivecset in self.ivecs:
            if ivecset.size() > 0:
                name = ivecset.name
                reg_name = re.sub('/.*', '', ivecset.name)
                Utils.mkdir_p(os.path.join(out_dir, os.path.dirname(name)))
                with open(os.path.join(out_dir, name + '.rttm'), 'w') as f:
                    for i, ivec in enumerate(ivecset.ivecs):
                        start, end = ivec.window_start, ivec.window_end
                        idx = np.argmax(scores[name].T[i])
                        f.write('SPEAKER {} 1 {} {} <NA> <NA> {}_spkr_{} <NA>\n'.format(
                            reg_name, float(start / 1000.0), float((end - start) / 1000.0), reg_name, idx))
            else:
                logwarning('[Diarization.dump_rttm] No i-vectors to dump in {}.'.format(ivecset.name))
