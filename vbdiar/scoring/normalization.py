#!/usr/bin/env python

import os

import h5py
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from sklearn.metrics.pairwise import cosine_similarity

from vbdiar.features.features import Features
from vbdiar.features.raw2ivec import RATE, get_num_frames
from vbdiar.scoring.plda import PLDA
from vbdiar.utils.utils import loginfo, logwarning, Utils


class Normalization(object):
    """ Speaker normalization S-Norm.

    """
    norm_ivec = None

    def __init__(self, norm_list, audio_dir=None, rttm_dir=None, in_ivec_dir=None, out_ivec_dir=None,
                 fea2ivec=None, plda=None, audio_suffix='wav', rttm_suffix='rttm'):
        """

        Args:
            norm_list (str): path to normalization list
            audio_dir (str|None): path to audio directory
            rttm_dir (str|None): path to directory with rttm files
            in_ivec_dir (str|None): path to directory with i-vectors
            out_ivec_dir (str|None): path to directory for storing i-vectors
            fea2ivec (Fea2Ivec|None): initialized Fea2Ivec object
            plda (PLDA|None): plda model object
            audio_suffix (str): suffix of wav files
            rttm_suffix (str): suffix of rttm files
        """
        self.audio_dir = audio_dir
        self.norm_list = norm_list
        self.rttm_dir = rttm_dir
        self.fea2ivec = fea2ivec
        self.plda = plda
        self.audio_suffix = audio_suffix
        self.rttm_suffix = rttm_suffix
        self.in_ivec_dir = in_ivec_dir
        self.out_ivec_dir = out_ivec_dir
        if self.in_ivec_dir is None:
            self.norm_ivec = self.extract_ivecs()
        else:
            self.norm_ivec = self.load_ivecs()

    def extract_ivecs(self):
        """ Extract normalization i-vectors using averaging.

            Returns:

        """
        speakers_dict = {}

        with open(self.norm_list, 'r') as f:
            for line in f:
                if len(line.split()) > 1:  # number of speakers is defined
                    line = line.split()[0]
                speakers_dict = self.process_file(line, speakers_dict)

        for dict_key in speakers_dict:
            speakers_dict[dict_key] = np.mean(speakers_dict[dict_key], axis=0)

        if self.out_ivec_dir:
            for speaker in speakers_dict:
                Utils.mkdir_p(os.path.join(self.out_ivec_dir, os.path.dirname(speaker)))
                h5file = h5py.File('{}.{}'.format(os.path.join(self.out_ivec_dir, speaker), 'h5'), 'w')
                h5file.create_dataset(speaker, data=speakers_dict[speaker])
                h5file.close()
        return np.array(speakers_dict.values())

    def process_file(self, file_name, speakers_dict):
        loginfo('Processing file {} ...'.format(file_name.split()[0]))
        wav = '{}.{}'.format(os.path.join(self.audio_dir, file_name), self.audio_suffix)
        rate, sig = read(wav)
        if len(sig.shape) != 1:
            raise ValueError('Expected mono as input audio.')
        if rate != RATE:
            logwarning('The input file is expected to be in 8000 Hz, got {} Hz instead, resampling.'.format(rate))
            sig = signal.resample(sig, RATE)

        fea_extractor = Features()
        fea = fea_extractor(sig)

        with open('{}.{}'.format(os.path.join(self.rttm_dir, file_name), self.rttm_suffix), 'r') as f:
            for line in f:
                start, dur, speaker = float(line.split()[3]) * 1000, float(line.split()[4]) * 1000, line.split()[7]
                end = start + dur
                start, end = get_num_frames(start), get_num_frames(end)
                w = self.fea2ivec.get_ivec(fea[start:end])
                if speaker not in speakers_dict.keys():
                    speakers_dict[speaker] = w
                else:
                    speakers_dict[speaker] = np.append(speakers_dict[speaker], w, axis=0)
        return speakers_dict

    def load_ivecs(self):
        """ Load normalization i-vectors, scale and shift files and also pretrained model.

            :returns: i-vectors
            :rtype: numpy.array
        """
        ivecs_list = []
        with open(self.norm_list, 'r') as f:
            for line in f:
                if len(line.split()) > 1:
                    line = line.split()[0]
                    loginfo('Loading h5 normalization file {} ...'.format(line))
                    h5file = h5py.File('{}.{}'.format(os.path.join(self.in_ivec_dir, line), 'h5'), 'r')
                    for h5_key in h5file.keys():
                        ivecs_list.append(h5file[h5_key][:].flatten())
        return np.array(ivecs_list)

    def s_norm(self, test, enroll):
        """ Run S-Norm on input i-vectors.

            :param test: test i-vectors
            :type test: numpy.array
            :param enroll: enroll i-vectors
            :type enroll: numpy.array
            :returns: scores matrix
            :rtype: numpy.array
        """
        if self.plda:
            a = self.plda.score(test, self.norm_ivec)
            b = self.plda.score(enroll, self.norm_ivec)
            c = self.plda.score(enroll, test)
        else:
            a = cosine_similarity(test, self.norm_ivec).T
            b = cosine_similarity(enroll, self.norm_ivec).T
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
