#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import os
import logging
import tempfile
import subprocess

from vbdiar.kaldi import nnet3bin_path
from vbdiar.kaldi.utils import write_txt_matrix, read_txt_vectors


logger = logging.getLogger(__name__)


class KaldiXVectorExtraction(object):

    def __init__(self, nnet, binary_path=nnet3bin_path, use_gpu=False,
                 min_chunk_size=25, chunk_size=10000, cache_capacity=64):
        """ Initialize Kaldi x-vector extractor.

        Args:
            nnet (string_types): path to neural net
            use_gpu (bool):
            min_chunk_size (int):
            chunk_size (int):
            cache_capacity (int):
        """
        self.nnet3_xvector_compute = os.path.join(binary_path, 'nnet3-xvector-compute')
        if not os.path.exists(self.nnet3_xvector_compute):
            raise ValueError(
                'Path to nnet3-xvector-compute - `{}` does not exists.'.format(self.nnet3_xvector_compute))
        self.nnet3_copy = os.path.join(binary_path, 'nnet3-copy')
        if not os.path.exists(self.nnet3_copy):
            raise ValueError(
                'Path to nnet3-copy - `{}` does not exists.'.format(self.nnet3_copy))
        if not os.path.isfile(nnet):
            raise ValueError('Invalid path to nnet `{}`.'.format(nnet))
        else:
            self.nnet = nnet
        self.binary_path = binary_path
        self.use_gpu = use_gpu
        self.min_chunk_size = min_chunk_size
        self.chunk_size = chunk_size
        self.cache_capacity = cache_capacity

    def features2embeddings(self, data_dict):
        """ Extract x-vector embeddings from feature vectors.

        Args:
            data_dict (Dict):

        Returns:

        """
        tmp_data_dict = {}
        for key in data_dict:
            tmp_data_dict[f'{key[0]}_{key[1]}'] = data_dict[key]
        with tempfile.NamedTemporaryFile() as xvec_ark, tempfile.NamedTemporaryFile() as mfcc_ark:
            write_txt_matrix(path=mfcc_ark.name, data_dict=tmp_data_dict)

            args = [self.nnet3_xvector_compute,
                    '--use-gpu={}'.format('yes' if self.use_gpu else 'no'),
                    '--min-chunk-size={}'.format(str(self.min_chunk_size)),
                    '--chunk-size={}'.format(str(self.chunk_size)),
                    '--cache-capacity={}'.format(str(self.cache_capacity)),
                    self.nnet, 'ark,t:{}'.format(mfcc_ark.name), 'ark,t:{}'.format(xvec_ark.name)]

            logger.info('Extracting x-vectors from {} feature vectors to `{}`.'.format(len(tmp_data_dict), xvec_ark.name))
            process = subprocess.Popen(
                args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.binary_path, shell=False)
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise ValueError('`{}` binary returned error code {}.{}{}'.format(
                    self.nnet3_xvector_compute, process.returncode, os.linesep, stderr))
            tmp_xvec_dict = read_txt_vectors(xvec_ark.name)
            xvec_dict = {}
            for key in tmp_xvec_dict:
                new_key = tuple(key.split('_'))
                xvec_dict[new_key] = tmp_xvec_dict[key]
            return xvec_dict
