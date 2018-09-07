#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import re
import pickle
import logging
from shutil import rmtree
from subprocess import check_output
from tempfile import mkdtemp, NamedTemporaryFile

import numpy as np
from spherecluster import SphericalKMeans
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import KMeans as sklearnKMeans
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.clustering.pldakmeans import PLDAKMeans
from vbdiar.scoring.normalization import Normalization
from vbdiar.utils import mkdir_p
from vbdiar.utils.utils import Utils


CDIR = os.path.dirname(os.path.realpath(__file__))
MD_EVAL_SCRIPT_PATH = os.path.join(CDIR, 'md-eval.pl')

logger = logging.getLogger(__name__)


def evaluate2rttms(reference_path, hypothesis_path, collar_size=0.25, evaluate_overlaps=False):
    """ Evaluate two rttms.

    Args:
        reference_path (string_types):
        hypothesis_path (string_types):
        collar_size (float):
        evaluate_overlaps (bool):

    Returns:
        float: diarization error rate
    """
    stdout = check_output([MD_EVAL_SCRIPT_PATH, '{}'.format('' if evaluate_overlaps else '-1'),
                           '-c', str(collar_size), '-r', reference_path, '-s', hypothesis_path])
    for line in stdout.split(os.linesep):
        if ' OVERALL SPEAKER DIARIZATION ERROR = ' in line:
            return float(line.replace(
                ' OVERALL SPEAKER DIARIZATION ERROR = ', '').replace(
                ' percent of scored speaker time  `(ALL)', ''))


def evaluate_all(reference_dir, hypothesis_dir, names, collar_size=0.25, evaluate_overlaps=False, rttm_ext='.rttm'):
    """ Evaluate all rttms in directories specified by list of names.

    Args:
        reference_dir (string_types): directory containing reference rttm files
        hypothesis_dir (string_types): directory containing hypothesis rttm files
        names (List[string_types]): list containing relative names
        collar_size (float):
        evaluate_overlaps (bool):
        rttm_ext (string_types): extension of rttm files

    Returns:
        float: diarization error rate
    """
    with NamedTemporaryFile() as ref, NamedTemporaryFile() as hyp:
        for name in names:
            with open('{}{}'.format(os.path.join(reference_dir, name), rttm_ext)) as f:
                for line in f:
                    ref.write(line)
                ref.write(os.linesep)
            with open('{}{}'.format(os.path.join(hypothesis_dir, name), rttm_ext)) as f:
                for line in f:
                    hyp.write(line)
                hyp.write(os.linesep)
            ref.flush()
            hyp.flush()

        return evaluate2rttms(ref.name, hyp.name, collar_size=collar_size, evaluate_overlaps=evaluate_overlaps)


class Diarization(object):
    """ Diarization class used as main diarization focused implementation.

    """
    def __init__(self, input_list, embeddings, embeddings_mean=None, lda=None, use_l2_norm=True, norm=None, plda=None):
        """ Initialize diarization class.

        Args:
            input_list (string_types): path to list of input files
            embeddings (string_types|List[EmbeddingSet]): path to directory containing embeddings or list
                of EmbeddingSet instances
            embeddings_mean (np.array):
            lda (np.array):
            use_l2_norm (bool):
            norm (Normalization): instance of class Normalization
            plda (PLDA): instance of class PLDA
        """
        self.input_list = input_list
        if isinstance(embeddings, str):
            self.embeddings_dir = embeddings
            self.embeddings = list(self.load_embeddings())
        else:
            self.embeddings = embeddings
        self.use_l2_norm = use_l2_norm
        self.norm = norm
        self.plda = plda

        for embedding_set in self.embeddings:
            for embedding in embedding_set:
                if embeddings_mean is not None:
                    embedding.data = embedding.data - embeddings_mean
                if lda is not None:
                    embedding.data = lda.dot(embedding.data)
                if use_l2_norm:
                    embedding.data = Utils.l2_norm(embedding.data[np.newaxis, :]).flatten()

    def get_embedding(self, name):
        """ Get embedding set by name.

        Args:
            name (string_types):

        Returns:
            EmbeddingSet:
        """
        for ii in self.embeddings:
            if name == ii.name:
                return ii
        raise ValueError('Name of the set not found - `{}`.'.format(name))

    def load_embeddings(self):
        """ Load embedding from pickled files.

        Returns:
            List[EmbeddingSet]:
        """
        with open(self.input_list, 'r') as f:
            for line in f:
                if len(line) > 0:
                    logger.info('Loading pickle file `{}.pkl` from `{}`.'.format(
                        line.rstrip().split()[0], self.embeddings_dir))
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
                            raise ValueError('Unexpected number of columns in input list `{}`.'.format(
                                self.input_list))
                    except IOError:
                        logger.warning('No pickle file found for `{}` in `{}`.'.format(
                            line.rstrip().split()[0], self.embeddings_dir))

    def score_embeddings(self, min_length, max_num_speakers):
        """ Score embeddings.

        Args:
            min_length (int): minimal length of segment used for clustering in miliseconds
            max_num_speakers (int): maximal number of speakers

        Returns:
            dict: dictionary with scores for each file
        """
        scores_dict = {}
        logger.info('Scoring using `{}`.'.format('PLDA' if self.plda is not None else 'cosine distance'))
        for embedding_set in self.embeddings:
            name = os.path.normpath(embedding_set.name)
            embeddings_all = embedding_set.get_all_embeddings()
            embeddings_long = embedding_set.get_longer_embeddings(min_length)
            if len(embeddings_long) == 0:
                logger.warning(
                    'No embeddings found longer than {} for embedding set `{}`.'.format(min_length, name))
                continue
            size = len(embedding_set)
            if size > 0:
                logger.info('Clustering `{}` using {} long embeddings.'.format(name, len(embeddings_long)))
                if embedding_set.num_speakers is not None:
                    num_speakers = embedding_set.num_speakers
                    if self.use_l2_norm:
                        kmeans_clustering = SphericalKMeans(
                            n_clusters=num_speakers, n_init=1000, n_jobs=1).fit(embeddings_long)
                    else:
                        kmeans_clustering = sklearnKMeans(
                            n_clusters=num_speakers, n_init=1000, n_jobs=1).fit(embeddings_long)
                    if self.plda is None:
                        centroids = kmeans_clustering.cluster_centers_
                    else:
                        centroids = PLDAKMeans(
                            kmeans_clustering.cluster_centers_, num_speakers, self.plda).fit(embeddings_long)
                else:
                    xm = xmeans(embeddings_long, kmax=max_num_speakers)
                    xm.process()
                    num_speakers = len(xm.get_clusters())
                    kmeans_clustering = sklearnKMeans(
                        n_clusters=num_speakers, n_init=100, n_jobs=1).fit(embeddings_long)
                    centroids = kmeans_clustering.cluster_centers_
                if self.norm is None:
                    if self.plda is None:
                        scores_dict[name] = cosine_similarity(embeddings_all, centroids).T
                    else:
                        scores_dict[name] = self.plda.score(embeddings_all, centroids)
                else:
                    scores_dict[name] = self.norm.s_norm(embeddings_all, centroids)
            else:
                logger.warning('No embeddings to score in `{}`.'.format(embedding_set.name))
        return scores_dict

    def dump_rttm(self, scores, out_dir):
        """ Dump rttm files to output directory. This function requires initialized embeddings.

        Args:
            scores (Dict): dictionary containing scores
            out_dir (string_types): path to output directory
        """
        for embedding_set in self.embeddings:
            if len(embedding_set) > 0:
                name = embedding_set.name
                reg_name = re.sub('/.*', '', embedding_set.name)
                mkdir_p(os.path.join(out_dir, os.path.dirname(name)))
                with open(os.path.join(out_dir, name + '.rttm'), 'w') as f:
                    for i, ivec in enumerate(embedding_set.embeddings):
                        start, end = ivec.window_start, ivec.window_end
                        idx = np.argmax(scores[name].T[i])
                        f.write('SPEAKER {} 1 {} {} <NA> <NA> {}_spkr_{} <NA>\n'.format(
                            reg_name, float(start / 1000.0), float((end - start) / 1000.0), reg_name, idx))
            else:
                logger.warning('No embedding to dump in {}.'.format(embedding_set.name))

    def evaluate(self, scores, in_rttm_dir, collar_size=0.25, evaluate_overlaps=False, rttm_ext='.rttm'):
        """ At first, separately evaluate each file based on ground truth segmentation. Then evaluate all files.

        Args:
            scores (dict): dictionary containing scores
            in_rttm_dir (string_types): input directory with rttm files
            collar_size (float): collar size for scoring
            evaluate_overlaps (bool): evaluate or ignore overlapping speech segments
            rttm_ext (string_types): extension for rttm files
        """
        tmp_dir = mkdtemp(prefix='rttm_')
        self.dump_rttm(scores, tmp_dir)
        for embedding_set in self.embeddings:
            name = embedding_set.name
            ground_truth_rttm = os.path.join(in_rttm_dir, '{}{}'.format(name, rttm_ext))
            if not os.path.exists(ground_truth_rttm):
                logger.warning('Ground truth rttm file not found in `{}`.'.format(ground_truth_rttm))
                continue
            # evaluate single rttm
            der = evaluate2rttms(ground_truth_rttm, os.path.join(tmp_dir, '{}{}'.format(name, rttm_ext)),
                                 collar_size=collar_size, evaluate_overlaps=evaluate_overlaps)
            logger.info('`{}` DER={}'.format(name, der))

        # evaluate all rttms
        der = evaluate_all(reference_dir=in_rttm_dir, hypothesis_dir=tmp_dir, names=scores.keys(),
                           collar_size=collar_size, evaluate_overlaps=evaluate_overlaps, rttm_ext=rttm_ext)
        logger.info('`Total` DER={}'.format(der))
        rmtree(tmp_dir)


