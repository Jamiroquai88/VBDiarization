#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import logging
import cPickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.scoring.plda import PLDA
from vbdiar.features.segments import get_num_frames
from vbdiar.embeddings.embedding import extract_embeddings


logger = logging.getLogger(__name__)


class Normalization(object):
    """ Speaker normalization S-Norm. """
    norm_embeddings = None

    def __init__(self, norm_list, audio_dir=None, in_rttm_dir=None, in_emb_dir=None, out_emb_dir=None, min_length=None,
                 features_extractor=None, embedding_extractor=None, plda=None, wav_suffix='.wav', rttm_suffix='.rttm'):
        """ Initialize normalization object.

        Args:
            norm_list (string_types): path to normalization list
            audio_dir (string_types|None): path to audio directory
            in_rttm_dir (string_types|None): path to directory with rttm files
            in_emb_dir (str|None): path to directory with i-vectors
            out_emb_dir (str|None): path to directory for storing embeddings
            min_length (int): minimal length for extracting embeddings
            features_extractor (Any): object for feature extraction
            embedding_extractor (Any): object for extracting embedding
            plda (PLDA|None): plda model object
            wav_suffix (string_types): suffix of wav files
            rttm_suffix (string_types): suffix of rttm files
        """
        if audio_dir:
            self.audio_dir = os.path.abspath(audio_dir)
        self.norm_list = norm_list
        if in_rttm_dir:
            self.in_rttm_dir = os.path.abspath(in_rttm_dir)
        self.features_extractor = features_extractor
        self.embedding_extractor = embedding_extractor
        self.plda = plda
        self.wav_suffix = wav_suffix
        self.rttm_suffix = rttm_suffix
        if in_emb_dir:
            self.in_emb_dir = os.path.abspath(in_emb_dir)
        if out_emb_dir:
            self.out_emb_dir = os.path.abspath(out_emb_dir)
        self.min_length = min_length
        if self.in_emb_dir is None:
            self.norm_embeddings = self.extract_embeddings()
        else:
            self.norm_embeddings = self.load_embeddings()

    def extract_embeddings(self):
        """ Extract normalization embeddings using averaging.

        Returns:
            Tuple[np.array, np.array]: vectors for individual speakers, global mean over all speakers
        """
        speakers_dict = {}
        with open(self.norm_list) as f:
            for line in f:
                if len(line.split()) > 1:  # number of speakers is defined
                    line = line.split()[0]
                else:
                    line = line.replace(os.linesep, '')
                speakers_dict = self.process_file(line, speakers_dict)

        for dict_key in speakers_dict:
            speakers_dict[dict_key] = np.mean(speakers_dict[dict_key], axis=0)

        if self.out_emb_dir:
            for speaker in speakers_dict:
                with open(os.path.join(self.out_emb_dir, '{}.pkl'.format(speaker)), 'wb') as f:
                    cPickle.dump(speakers_dict[speaker], f, cPickle.HIGHEST_PROTOCOL)
        speaker_embeddings = np.array(speakers_dict.values())
        return speaker_embeddings, np.mean(speaker_embeddings, axis=0)

    def process_file(self, file_name, speakers_dict):
        """ Extract embeddings for all defined speakers.

        Args:
            file_name (string_types): path to input audio file
            speakers_dict (dict): dictionary containing all embedding across speakers

        Returns:
            dict: updated dictionary with speakers
        """
        logger.info('Processing file `{}` ...'.format(file_name.split()[0]))
        # extract features from whole audio
        _, features = self.features_extractor.audio2features(
            os.path.join(self.audio_dir, '{}{}'.format(file_name, self.wav_suffix)))

        # process utterances of the speakers
        features_dict = {}
        with open('{}{}'.format(os.path.join(self.in_rttm_dir, file_name), self.rttm_suffix)) as f:
            for line in f:
                start, dur, speaker = float(line.split()[3]) * 1000, float(line.split()[4]) * 1000, line.split()[7]
                end = start + dur
                start, end = get_num_frames(int(start)), get_num_frames(int(end))
                if speaker not in features_dict:
                    features_dict[speaker] = {}
                features_dict[speaker]['{}_{}'.format(start, end)] = features[start:end]
        for speaker in features_dict:
            embedding_set = extract_embeddings(features_dict[speaker], self.embedding_extractor)
            embeddings_long = embedding_set.get_longer_embeddings(min_length=self.min_length)
            if speaker not in speakers_dict.keys():
                speakers_dict[speaker] = embeddings_long
            else:
                speakers_dict[speaker] = np.append(speakers_dict[speaker], embeddings_long, axis=0)
        return speakers_dict

    def load_embeddings(self):
        """ Load normalization embeddings from pickle files.

        Returns:
            np.array: embeddings per speaker
        """
        embeddings = []
        with open(self.norm_list) as f:
            for line in f:
                if len(line.split()) > 1:  # number of speakers is defined
                    line = line.split()[0]
                embedding_path = os.path.join(self.in_emb_dir, '{}.pkl'.format(line))
                if os.path.isfile(embedding_path):
                    with open(embedding_path) as fp:
                        embeddings.append(cPickle.load(fp))
                else:
                    logger.warning('No pickle file found for `{}` in `{}`.'.format(line, self.in_emb_dir))
        return np.array(embeddings)

    def s_norm(self, test, enroll):
        """ Run speaker normalization (S-Norm) on cached embeddings.

        Args:
            test (np.array): test embedding
            enroll (np.array): enroll embedding

        Returns:
            float: hypothesis
        """
        if self.plda:
            a = self.plda.score(test, self.norm_embeddings)
            b = self.plda.score(enroll, self.norm_embeddings)
            c = self.plda.score(enroll, test)
        else:
            a = cosine_similarity(test, self.norm_embeddings).T
            b = cosine_similarity(enroll, self.norm_embeddings).T
            c = cosine_similarity(enroll, test).T
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
