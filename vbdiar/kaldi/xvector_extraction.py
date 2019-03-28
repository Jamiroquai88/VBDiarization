#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import os

import onnxruntime


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
        xvec_dict = {}
        for name in data_dict:
            xvec = self.sess.run(None, {self.input_name: data_dict[name]})
            xvec_dict[name] = xvec
        return xvec_dict
