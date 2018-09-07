#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import pickle
import numpy as np

from vbdiar.utils import mkdir_p


def extract_embeddings(features_dict, embedding_extractor):
    """ Extract embeddings from multiple segments.

    Args:
        features_dict (Dict): dictionary with segment range as key and features as values
        embedding_extractor (Any):

    Returns:
        EmbeddingSet: extracted embedding in embedding set
    """
    embedding_set = EmbeddingSet()
    embeddings = embedding_extractor.features2embeddings(features_dict)
    for embedding_key in embeddings:
        start, end = embedding_key.split('_')
        embedding_set.add(embeddings[embedding_key], window_start=int(start), window_end=int(end))
    return embedding_set


class Embedding(object):
    """ Class for basic i-vector operations.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.data = None
        self.features = None
        self.window_start = None
        self.window_end = None


class EmbeddingSet(object):
    """ Class for encapsulating ivectors set.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.name = None
        self.num_speakers = None
        self.embeddings = []

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

    def get_all_embeddings(self):
        """ Get all ivectors.

        """
        a = []
        for i in self.embeddings:
            a.append(i.data.flatten())
        return np.array(a)

    def get_longer_embeddings(self, min_length):
        """ Get i-vectors extracted from longer segments than minimal length.

        Args:
            min_length (int): minimal length of segment in miliseconds

        Returns:
            np.array: i-vectors
        """
        a = []
        for ivec in self.embeddings:
            if ivec.window_end - ivec.window_start >= min_length:
                a.append(ivec.data.flatten())
        return np.array(a)

    def add(self, data, window_start, window_end, mfccs=None):
        """ Add ivector to set.

            :param data: i-vector data
            :type data: numpy.array
            :param window_start: start of the window [ms]
            :type window_start: int
            :param window_end: end of the window [ms]
            :type window_end: int
        """
        i = Embedding()
        i.data = data
        i.window_start = window_start
        i.window_end = window_end
        i.features = mfccs
        self.__append(i)

    def __append(self, ivec):
        """ Append ivec to set of ivecs.

            :param ivec: input ivector
            :type ivec: Embedding
        """
        ii = 0
        for vp in self.embeddings:
            if vp.window_start > ivec.window_start:
                break
            ii += 1
        self.embeddings.insert(ii, ivec)

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
