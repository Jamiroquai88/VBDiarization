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

import kaldiio

from vbdiar.kaldi import featbin_path


logger = logging.getLogger(__name__)


class KaldiMFCCFeatureExtraction(object):

    def __init__(self, config_path, binary_path=featbin_path, apply_cmvn_sliding=True,
                 norm_vars=False, center=True, cmn_window=300):
        """ Initialize Kaldi MFCC extraction component. Names of the arguments keep original Kaldi convention.

        Args:
            config_path (string_types): path to config file
            binary_path (string_types): path to directory containing binaries
            apply_cmvn_sliding (bool): apply cepstral mean and variance normalization
            norm_vars (bool): normalize variances
            center (bool): center window
            cmn_window (int): window size
        """
        self.binary_path = binary_path
        self.config_path = config_path
        self.apply_cmvn_sliding = apply_cmvn_sliding
        self.norm_vars = norm_vars
        self.center = center
        self.cmn_window = cmn_window
        self.compute_mfcc_feats_bin = os.path.join(binary_path, 'compute-mfcc-feats')
        if not os.path.exists(self.compute_mfcc_feats_bin):
            raise ValueError('Path to compute-mfcc-feats - {} does not exists.'.format(self.compute_mfcc_feats_bin))
        self.copy_feats_bin = os.path.join(binary_path, 'copy-feats')
        if not os.path.exists(self.copy_feats_bin):
            raise ValueError('Path to copy-feats - {} does not exists.'.format(self.copy_feats_bin))
        self.apply_cmvn_sliding_bin = os.path.join(binary_path, 'apply-cmvn-sliding')
        if not os.path.exists(self.apply_cmvn_sliding_bin):
            raise ValueError('Path to apply-cmvn-sliding - {} does not exists.'.format(self.apply_cmvn_sliding_bin))

    def __str__(self):
        return '<mfcc_config={}>'.format(self.config_path)

    def audio2features(self, input_path):
        """ Extract features from list of files into list of numpy.arrays

        Args:
            input_path (string_types): audio file path

        Returns:
            Tuple[str, np.array]: path to Kaldi ark file containing features and features itself
        """
        with tempfile.NamedTemporaryFile(mode='w') as wav_scp, tempfile.NamedTemporaryFile() as mfcc_ark:
            # dump list of file to wav.scp file
            wav_scp.write('{} {}{}'.format(input_path, input_path, os.linesep))
            wav_scp.flush()

            # run fextract
            args = [self.compute_mfcc_feats_bin, f'--config={self.config_path}', f'scp:{wav_scp.name}',
                    f'ark:{mfcc_ark.name if not self.apply_cmvn_sliding else "-"}']
            logger.info('Extracting MFCC features from `{}`.'.format(input_path))
            compute_mfcc_feats = subprocess.Popen(
                args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.binary_path, shell=False)
            if not self.apply_cmvn_sliding:
                # do not apply cmvn, so just simply compute features
                _, stderr = compute_mfcc_feats.communicate()
                if compute_mfcc_feats.returncode != 0:
                    raise ValueError(f'`{self.compute_mfcc_feats_bin}` binary returned error code '
                                     f'{compute_mfcc_feats.returncode}.{os.linesep}{stderr}')
            else:
                args2 = [self.apply_cmvn_sliding_bin, f'--norm-vars={str(self.norm_vars).lower()}',
                         f'--center={str(self.center).lower()}', f'--cmn-window={str(self.cmn_window)}',
                         'ark:-', f'ark:{mfcc_ark.name}']
                apply_cmvn_sliding = subprocess.Popen(args2, stdin=compute_mfcc_feats.stdout,
                                                      stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
                _, stderr = apply_cmvn_sliding.communicate()
                if apply_cmvn_sliding.returncode == 0:
                    pass
                else:
                    raise ValueError(f'`{self.compute_mfcc_feats_bin}` binary returned error code '
                                     f'{compute_mfcc_feats.returncode}.{os.linesep}{stderr}')
                ark = kaldiio.load_ark(mfcc_ark.name)
                for key, numpy_array in ark:
                    return numpy_array
