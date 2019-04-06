#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import logging
import pickle
import multiprocessing

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.features.segments import get_frames_from_time
from vbdiar.embeddings.embedding import extract_embeddings
from vbdiar.utils import mkdir_p
from vbdiar.utils.utils import Utils

logger = logging.getLogger(__name__)


def process_files(fns, speakers_dict, features_extractor, embedding_extractor,
                  audio_dir, wav_suffix, in_rttm_dir, rttm_suffix, min_length, n_jobs=1):
    """

    Args:
        fns:
        speakers_dict:
        features_extractor:
        embedding_extractor:
        audio_dir:
        wav_suffix:
        in_rttm_dir:
        rttm_suffix:
        min_length:
        n_jobs:

    Returns:

    """
    kwargs = dict(speakers_dict=speakers_dict, features_extractor=features_extractor,
                  embedding_extractor=embedding_extractor, audio_dir=audio_dir, wav_suffix=wav_suffix,
                  in_rttm_dir=in_rttm_dir, rttm_suffix=rttm_suffix, min_length=min_length)
    if n_jobs == 1:
        ret = _process_files((fns, kwargs))
    else:
        pool = multiprocessing.Pool(n_jobs)
        ret = pool.map(_process_files, ((part, kwargs) for part in Utils.partition(fns, n_jobs)))
    return ret


def _process_files(dargs):
    """

    Args:
        dargs:

    Returns:

    """
    fns, kwargs = dargs
    ret = []
    for fn in fns:
        ret.append(process_file(file_name=fn, **kwargs))
    return ret


def process_file(file_name, speakers_dict, features_extractor, embedding_extractor,
                 audio_dir, wav_suffix, in_rttm_dir, rttm_suffix, min_length):
    """ Extract embeddings for all defined speakers.

    Args:
        file_name (string_types): path to input audio file
        speakers_dict (dict): dictionary containing all embedding across speakers
        features_extractor (Any):
        embedding_extractor (Any):
        audio_dir (string_types):
        wav_suffix (string_types):
        in_rttm_dir (string_types):
        rttm_suffix (string_types):
        min_length (float):

    Returns:
        dict: updated dictionary with speakers
    """
    logger.info('Processing file `{}`.'.format(file_name.split()[0]))
    # extract features from whole audio
    features = features_extractor.audio2features(os.path.join(audio_dir, '{}{}'.format(file_name, wav_suffix)))

    # process utterances of the speakers
    features_dict = {}
    with open(f'{os.path.join(in_rttm_dir, file_name)}{rttm_suffix}') as f:
        for line in f:
            start_time, dur = int(float(line.split()[3]) * 1000), int(float(line.split()[4]) * 1000)
            speaker = line.split()[7]
            if dur > min_length:
                end_time = start_time + dur
                start, end = get_frames_from_time(int(start_time)), get_frames_from_time(int(end_time))
                if speaker not in features_dict:
                    features_dict[speaker] = {}
                
                assert 0 <= start < end, \
                    f'Incorrect timing for extracting features, start: {start}, size: {features.shape[0]}, end: {end}.'
                if end >= features.shape[0]:
                    end = features.shape[0] - 1
                features_dict[speaker][(start_time, end_time)] = features[start:end]
    for speaker in features_dict:
        embedding_set = extract_embeddings(features_dict[speaker], embedding_extractor)
        embeddings_long = embedding_set.get_all_embeddings()
        if speaker not in speakers_dict.keys():
            speakers_dict[speaker] = embeddings_long
        else:
            speakers_dict[speaker] = np.concatenate((speakers_dict[speaker], embeddings_long), axis=0)
    return speakers_dict


class Normalization(object):
    """ Speaker normalization S-Norm. """
    embeddings = None
    in_emb_dir = None

    def __init__(self, norm_list, audio_dir=None, in_rttm_dir=None, in_emb_dir=None,
                 out_emb_dir=None, min_length=None, features_extractor=None, embedding_extractor=None,
                 plda=None, wav_suffix='.wav', rttm_suffix='.rttm', n_jobs=1):
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
        else:
            raise ValueError('It is required to have input rttm files for normalization.')
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
        self.n_jobs = n_jobs
        if self.in_emb_dir is None:
            self.embeddings = self.extract_embeddings()
        else:
            self.embeddings = self.load_embeddings()
        self.mean = np.mean(self.embeddings, axis=0)

    def __iter__(self):
        current = 0
        while current < len(self.embeddings):
            yield self.embeddings[current]
            current += 1

    def __getitem__(self, key):
        return self.embeddings[key]

    def __setitem__(self, key, value):
        self.embeddings[key] = value

    def __len__(self):
        return len(self.embeddings)

    def extract_embeddings(self):
        """ Extract normalization embeddings using averaging.

        Returns:
            Tuple[np.array, np.array]: vectors for individual speakers, global mean over all speakers
        """
        speakers_dict, fns = {}, []
        with open(self.norm_list) as f:
            for line in f:
                if len(line.split()) > 1:  # number of speakers is defined
                    line = line.split()[0]
                else:
                    line = line.replace(os.linesep, '')
                fns.append(line)

        speakers_dict = process_files(fns, speakers_dict=speakers_dict, features_extractor=self.features_extractor,
                                      embedding_extractor=self.embedding_extractor, audio_dir=self.audio_dir,
                                      wav_suffix=self.wav_suffix, in_rttm_dir=self.in_rttm_dir,
                                      rttm_suffix=self.rttm_suffix, min_length=self.min_length, n_jobs=self.n_jobs)
        assert len(speakers_dict) == len(fns)
        # all are the same
        merged_speakers_dict = speakers_dict[0]

        if self.out_emb_dir:
            for speaker in merged_speakers_dict:
                out_path = os.path.join(self.out_emb_dir, f'{speaker}.pkl')
                mkdir_p(os.path.dirname(out_path))
                with open(out_path, 'wb') as f:
                    pickle.dump(merged_speakers_dict[speaker], f, pickle.HIGHEST_PROTOCOL)

        for speaker in merged_speakers_dict:
            merged_speakers_dict[speaker] = np.mean(merged_speakers_dict[speaker], axis=0)

        return np.array(list(merged_speakers_dict.values()))

    def load_embeddings(self):
        """ Load normalization embeddings from pickle files.

        Returns:
            np.array: embeddings per speaker
        """
        embeddings, speakers = [], set()
        with open(self.norm_list) as f:
            for file_name in f:
                if len(file_name.split()) > 1:  # number of speakers is defined
                    file_name = file_name.split()[0]
                else:
                    file_name = file_name.replace(os.linesep, '')
                with open('{}{}'.format(os.path.join(self.in_rttm_dir, file_name), self.rttm_suffix)) as fp:
                    for line in fp:
                        speakers.add(line.split()[7])

        logger.info('Loading pickled normalization embeddings from `{}`.'.format(self.in_emb_dir))
        for speaker in speakers:
            embedding_path = os.path.join(self.in_emb_dir, '{}.pkl'.format(speaker))
            if os.path.isfile(embedding_path):
                logger.info('Loading normalization pickle file `{}`.'.format(speaker))
                with open(embedding_path, 'rb') as f:
                    # append mean from speaker's embeddings
                    speaker_embeddings = pickle.load(f)
                    embeddings.append(np.mean(speaker_embeddings, axis=0))
            else:
                logger.warning('No pickle file found for `{}` in `{}`.'.format(speaker, self.in_emb_dir))
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
            a = self.plda.score(test, self.embeddings).T
            b = self.plda.score(enroll, self.embeddings).T
            c = self.plda.score(enroll, test).T
        else:
            a = cosine_similarity(test, self.embeddings).T
            b = cosine_similarity(enroll, self.embeddings).T
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
        return np.array(scores)
