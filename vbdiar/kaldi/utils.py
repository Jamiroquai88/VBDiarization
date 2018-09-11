#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import os

import numpy as np


def read_txt_matrix(path):
    """ Read features file in text format. This code expects correct format of input file.

    Args:
        path (string_types): path to txt file

    Returns:
        Dict[np.array]: name to array mapping
    """
    data_dict, name = {}, None
    with open(path) as f:
        for line in f:
            if '[' in line:
                name = line.split()[0] if len(line) > 3 else ''
                continue
            elif ']' in line:
                line = line.replace(' ]', '')
            assert name is not None, 'Incorrect format of input file `{}`.'.format(path)
            if name not in data_dict:
                data_dict[name] = []
            data_dict[name].append(np.fromstring(line, sep=' ', dtype=np.float32))
    for name in data_dict:
        data_dict[name] = np.array(data_dict[name])
    return data_dict


def write_txt_matrix(path, data_dict):
    """ Write features into file in text format. This code expects correct format of input dictionary.

    Args:
        path (string_types): path to txt file
        data_dict (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict.keys()):
            f.write('{}  ['.format(name, os.linesep))
            for row_idx in range(data_dict[name].shape[0]):
                f.write('{}  '.format(os.linesep))
                data_dict[name][row_idx].tofile(f, sep=' ', format='%.6f')
            f.write(' ]{}'.format(os.linesep))


def read_txt_vectors(path):
    """ Read vectors file in text format. This code expects correct format of input file.

        Args:
            path (string_types): path to txt file

        Returns:
            Dict[np.array]: name to array mapping
    """
    data_dict = {}
    with open(path) as f:
        for line in f:
            splitted_line = line.split()
            name = splitted_line[0]
            end_idx = splitted_line.index(']')
            vector_data = np.array([float(single_float) for single_float in splitted_line[2:end_idx]])
            data_dict[name] = vector_data
    return data_dict
