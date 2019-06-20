#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import logging
import os

import numpy as np
import onnxruntime


logger = logging.getLogger(__name__)

MIN_SIGNAL_LEN = 25


class ONNXXVectorExtraction(object):

    def __init__(self, onnx_path):
        """ Initialize ONNX x-vector extractor.

        Args:
            onnx_path (str): path to neural net in ONNX format, see https://github.com/onnx/onnx
        """
        if not os.path.isfile(onnx_path):
            raise ValueError(f'Invalid path to nnet `{onnx_path}`.')
        else:
            self.onnx_path = onnx_path
            self.sess = onnxruntime.InferenceSession(onnx_path)
            self.input_name = self.sess.get_inputs()[0].name

    def features2embeddings(self, data_dict):
        """ Extract x-vector embeddings from feature vectors.

        Args:
            data_dict (Dict):

        Returns:

        """
        logger.info(f'Extracting x-vectors from {len(data_dict)} segments.')
        xvec_dict = {}
        for name in data_dict:
            signal_len, num_coefs = data_dict[name].shape
            # here we need to avoid failing on very short inputs, so we will just concatenate frames in time
            if signal_len == 0:
                continue
            elif signal_len < MIN_SIGNAL_LEN:
                for i in range(MIN_SIGNAL_LEN // signal_len):
                    data_dict[name] = np.concatenate((data_dict[name], data_dict[name]), axis=0)
            xvec = self.sess.run(None, {self.input_name: data_dict[name].T[np.newaxis, :, :]})[0]
            xvec_dict[name] = xvec.squeeze()
        return xvec_dict
