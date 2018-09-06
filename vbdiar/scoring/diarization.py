#!/usr/bin/env python

import os
import re
import pickle
import logging

import numpy as np
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.clustering.pldakmeans import PLDAKMeans
from vbdiar.scoring.normalization import Normalization
from vbdiar.utils import mkdir_p
from vbdiar.utils.utils import Utils


logger = logging.getLogger(__name__)


class Diarization(object):
    """ Diarization class used as main diarization focused implementation.

    """
    def __init__(self, input_list, embeddings, embeddings_mean=None, norm=None, plda=None):
        """ Initialize diarization class.

        Args:
            input_list (string_types): path to list of input files
            embeddings (string_types|List[EmbeddingSet]): path to directory containing embeddings or list
                of EmbeddingSet instances
            embeddings_mean (np.array):
            norm (Normalization): instance of class Normalization
            plda (PLDA): instance of class PLDA
        """
        self.input_list = input_list
        if isinstance(embeddings, str):
            self.embeddings_dir = embeddings
            self.embeddings = list(self.load_embeddings())
            for emb_set_idx in range(len(self.embeddings)):
                for emb_idx in range(len(self.embeddings[emb_set_idx])):
                    if embeddings_mean is not None:
                        self.embeddings[emb_set_idx][emb_idx].data = \
                            self.embeddings[emb_set_idx][emb_idx].data - embeddings_mean
        else:
            self.embeddings = embeddings
        self.norm = norm
        self.plda = plda

    def get_embedding(self, name):
        """

        Args:
            name:

        Returns:

        """
        for ii in self.embeddings:
            print ii.name
            if name == ii.name:
                return ii
        raise ValueError('Name of the set not found - {}.'.format(name))

    def load_embeddings(self):
        """

        Returns:

        """
        with open(self.input_list, 'r') as f:
            for line in f:
                logger.info('Loading pickle file `{}`.'.format(line.rstrip().split()[0]))
                line = line.rstrip()
                try:
                    if len(line.split()) == 1:
                        with open(os.path.join(self.embeddings_dir, line + '.pkl')) as i:
                            yield pickle.load(i)
                    elif len(line.split()) == 2:
                        file_name = line.split()[0]
                        num_spks = int(line.split()[1])
                        with open(os.path.join(self.embeddings_dir, file_name + '.pkl')) as i:
                            ivec_set = pickle.load(i)
                            ivec_set.num_speakers = num_spks
                            yield ivec_set
                    else:
                        raise ValueError('Unexpected number of columns in input list `{}`.'.format(self.input_list))
                except IOError:
                    logger.warning('No pickle file found for `{}`.'.format(line.rstrip().split()[0]))

    def score_embeddings(self, min_length, max_num_speakers, num_threads, use_l2_norm=True):
        """ Score embeddings.

        Args:
            min_length (int): minimal length of segment used for clustering in miliseconds
            max_num_speakers (int): maximal number of speakers
            num_threads (int): number of threads to use

        Returns:
            dict: dictionary with scores for each file
        """
        scores_dict = {}
        for embedding_set in self.embeddings:
            name = os.path.normpath(embedding_set.name)
            embeddings_all = embedding_set.get_all_embeddings()
            ivecs_long = embedding_set.get_longer(min_length)
            logger.info('Scoring `{}` using `{}`'.format(name, 'PLDA' if self.plda is not None else 'cosine distance'))
            size = len(embedding_set)
            if size > 0:
                if embedding_set.num_speakers is not None:
                    num_speakers = embedding_set.num_speakers
                    sklearnkmeans = sklearnKMeans(
                        n_clusters=num_speakers, n_init=100, n_jobs=num_threads).fit(ivecs_long)
                    if self.plda is None:
                        centroids = sklearnkmeans.cluster_centers_
                    else:
                        centroids = PLDAKMeans(sklearnkmeans.cluster_centers_, num_speakers, self.plda).fit(ivecs_long)
                else:
                    xm = xmeans(ivecs_long, kmax=max_num_speakers)
                    xm.process()
                    num_speakers = len(xm.get_clusters())
                    sklearnkmeans = sklearnKMeans(
                        n_clusters=num_speakers, n_init=100, n_jobs=num_threads).fit(ivecs_long)
                    centroids = sklearnkmeans.cluster_centers_
                if self.norm is None:
                    if self.plda is None:
                        if use_l2_norm:
                            embeddings_all = Utils.l2_norm(embeddings_all)
                            centroids = Utils.l2_norm(centroids)
                        scores_dict[name] = cosine_similarity(embeddings_all, centroids).T
                    else:
                        scores_dict[name] = self.plda.score(embeddings_all, centroids)
                else:
                    if use_l2_norm:
                        embeddings_all = Utils.l2_norm(embeddings_all)
                        centroids = Utils.l2_norm(centroids)
                    scores_dict[name] = self.norm.s_norm(embeddings_all, centroids)
            else:
                logger.warning('No embeddings to score in `{}`.'.format(embedding_set.name))
        return scores_dict

    def dump_rttm(self, scores, out_dir):
        """

        Args:
            scores:
            out_dir:

        Returns:

        """
        for ivecset in self.embeddings:
            if ivecset.size() > 0:
                name = ivecset.name
                reg_name = re.sub('/.*', '', ivecset.name)
                mkdir_p(os.path.join(out_dir, os.path.dirname(name)))
                with open(os.path.join(out_dir, name + '.rttm'), 'w') as f:
                    for i, ivec in enumerate(ivecset.ivecs):
                        start, end = ivec.window_start, ivec.window_end
                        idx = np.argmax(scores[name].T[i])
                        f.write('SPEAKER {} 1 {} {} <NA> <NA> {}_spkr_{} <NA>\n'.format(
                            reg_name, float(start / 1000.0), float((end - start) / 1000.0), reg_name, idx))
            else:
                logger.warning('[Diarization.dump_rttm] No i-vectors to dump in {}.'.format(ivecset.name))
