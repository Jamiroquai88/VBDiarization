#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import os
import errno


def mkdir_p(path):
    """ Behaviour similar to mkdir -p in shell.

    Args:
        path (string_types): path to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise ValueError('Can not create directory {}.'.format(path))



