#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import urllib
import tarfile
import tempfile
from shutil import copyfile
from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

import massedit

from vbdiar.utils import mkdir_p
from vbdiar.kaldi import KALDI_ROOT_PATH

CDIR = os.path.dirname(os.path.realpath(__file__))
XVEC_MODELS_DIR = os.path.join(CDIR, 'models', 'x-vectors')


def install_scripts(directory):
    """ Call cmd commands to install extra software/repositories.

    Args:
        directory (str): path

    """
    if KALDI_ROOT_PATH is None or not os.path.isdir(KALDI_ROOT_PATH):
        raise ValueError('Please, set path to correct kaldi installation.')
    nnet_copy_binary = os.path.join(KALDI_ROOT_PATH, 'src', 'nnet3bin', 'nnet3-copy')
    if not os.path.isfile(nnet_copy_binary):
        raise ValueError('nnet3-copy binary not found in {}.'.format(os.path.dirname(nnet_copy_binary)))
    mkdir_p(XVEC_MODELS_DIR)
    with tempfile.NamedTemporaryFile() as f:
        urllib.urlretrieve(
            'http://kaldi-asr.org/models/0003_sre16_v2_1a.tar.gz', f.name)
        tar = tarfile.open(os.path.join(f.name), 'r:gz')
        tar.extractall(XVEC_MODELS_DIR)
        tar.close()

    nnet_raw_path = os.path.join(XVEC_MODELS_DIR, '0003_sre16_v2_1a', 'exp', 'xvector_nnet_1a', 'final.raw')

    # replace input of the last layer, so we can easily extract xvectors
    massedit.edit_files([nnet_raw_path], ["re.sub("
                                          "r'output-node name=output input=output.log-softmax objective=linear', "
                                          "r'output-node name=output input=tdnn6.affine objective=linear', line)"])

    copyfile(os.path.join(XVEC_MODELS_DIR, '0003_sre16_v2_1a', 'conf', 'mfcc.conf'),
             os.path.join(CDIR, 'configs', 'mfcc.conf'))


class PostDevelopCommand(develop):
    """ Post-installation for development mode."""
    def run(self):
        develop.run(self)
        self.execute(install_scripts, (self.egg_path,), msg='Running post install scripts')


class PostInstallCommand(install):
    """ Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self.execute(install_scripts, (self.install_lib,), msg='Running post install scripts')


setup(
    name='vbdiar',
    version='0.1',
    packages=find_packages(),
    url='',
    license='',
    author='profant',
    author_email='xprofa00@stud.fit.vutbr.cz',
    description='',
    cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand}
)
