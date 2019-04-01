#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import sys
import ctypes
import logging
import argparse
import multiprocessing

import numpy as np

from vbdiar.vad import get_vad
from vbdiar.utils import mkdir_p
from vbdiar.utils.utils import Utils
from vbdiar.embeddings.embedding import extract_embeddings
from vbdiar.scoring.diarization import Diarization
from vbdiar.scoring.normalization import Normalization
from vbdiar.kaldi.onnx_xvector_extraction import ONNXXVectorExtraction
from vbdiar.features.segments import get_segments, get_time_from_frames, get_frames_from_time
from vbdiar.kaldi.mfcc_features_extraction import KaldiMFCCFeatureExtraction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))


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


def process_files(fns, wav_dir, vad_dir, out_dir, features_extractor, embedding_extractor, min_size,
                  max_size, overlap, tolerance, wav_suffix='.wav', vad_suffix='.lab.gz', n_jobs=1):
    """ Process all files from list.

    Args:
        fns (list): name of files to process
        wav_dir (str): directory with wav files
        vad_dir (str): directory with vad files
        out_dir (str|None): output directory
        features_extractor (Any): intialized object for feature extraction
        embedding_extractor (Any): initialized object for embedding extraction
        max_size (int): maximal size of window in ms
        min_size (int): minimal size of window in ms
        overlap (int): size of window overlap in ms
        tolerance (int): accept given number of frames as speech even when it is marked as silence
        wav_suffix (str): suffix of wav files
        vad_suffix (str): suffix of vad files
        n_jobs (int): number of jobs to run in parallel

    Returns:
        List[EmbeddingSet]
    """
    kwargs = dict(wav_dir=wav_dir, vad_dir=vad_dir, out_dir=out_dir, features_extractor=features_extractor,
                  embedding_extractor=embedding_extractor, tolerance=tolerance, min_size=min_size,
                  max_size=max_size, overlap=overlap, wav_suffix=wav_suffix, vad_suffix=vad_suffix)
    if n_jobs == 1:
        ret = _process_files((fns, kwargs))
    else:
        pool = multiprocessing.Pool(n_jobs)
        ret = pool.map(_process_files, ((part, kwargs) for part in Utils.partition(fns, n_jobs)))
    return [item for sublist in ret for item in sublist]


def process_file(wav_dir, vad_dir, out_dir, file_name, features_extractor, embedding_extractor,
                 min_size, max_size, overlap, tolerance, wav_suffix='.wav', vad_suffix='.lab.gz'):
    """ Process single audio file.

    Args:
        wav_dir (str): directory with wav files
        vad_dir (str): directory with vad files
        out_dir (str): output directory
        file_name (str): name of the file
        features_extractor (Any): intialized object for feature extraction
        embedding_extractor (Any): initialized object for embedding extraction
        max_size (int): maximal size of window in ms
        max_size (int): maximal size of window in ms
        overlap (int): size of window overlap in ms
        tolerance (int): accept given number of frames as speech even when it is marked as silence
        wav_suffix (str): suffix of wav files
        vad_suffix (str): suffix of vad files

    Returns:
        EmbeddingSet
    """
    logger.info('Processing file {}.'.format(file_name.split()[0]))
    num_speakers = None
    if len(file_name.split()) > 1:  # number of speakers is defined
        file_name, num_speakers = file_name.split()[0], int(file_name.split()[1])

    wav_dir, vad_dir = os.path.abspath(wav_dir), os.path.abspath(vad_dir)
    if out_dir:
        out_dir = os.path.abspath(out_dir)

    # extract features
    _, features = features_extractor.audio2features(os.path.join(wav_dir, f'{file_name}{wav_suffix}'))

    # load voice activity detection from file
    vad, _, _ = get_vad(f'{os.path.join(vad_dir, file_name)}{vad_suffix}', features.shape[0])

    # parse segments and split features
    features_dict = {}
    for seg in get_segments(vad, max_size, tolerance):
        seg_start, seg_end = seg
        start, end = get_time_from_frames(seg_start), get_time_from_frames(seg_end)
        if start >= overlap:
            seg_start = get_frames_from_time(start - overlap)
        if seg_start > features.shape[0] - 1 or seg_end > features.shape[0] - 1:
            raise ValueError('Unexpected features dimensionality - check VAD input or audio.')
        features_dict[(start, end)] = features[seg_start:seg_end]

    # extract embedding for each segment
    embedding_set = extract_embeddings(features_dict, embedding_extractor)
    embedding_set.name = file_name
    embedding_set.num_speakers = num_speakers

    # save embeddings if required
    if out_dir is not None:
        mkdir_p(os.path.join(out_dir, os.path.dirname(file_name)))
        embedding_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))

    return embedding_set


def set_mkl(num_cores=1):
    """ Set number of cores for mkl library.

    Args:
        num_cores (int): number of cores
    """
    try:
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_rt.mkl_set_dynamic(ctypes.byref(ctypes.c_int(0)))
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_cores)))
    except OSError:
        logger.warning('Failed to import libmkl_rt.so, it will not be possible to use mkl backend.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract embeddings used for diarization from audio wav files.')

    # required
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', required=True)
    parser.add_argument('-c', '--configuration', help='input configuration of models',
                        action='store', required=True)
    parser.add_argument('-m', '--mode', required=True, choices=['sre', 'diarization'],
                        help='mode used - there are two possible modes, classic `diarization` mode which should'
                             'utterance into speakers and `sre` mode used for speaker recognition, which '
                             'runs clustering for N iterations and saves all clusters')

    # not required
    parser.add_argument('--audio-dir',
                        help='directory with audio files in .wav format - 8000Hz, 16bit-s, 1c', required=False)
    parser.add_argument('--vad-dir',
                        help='directory with lab files - Voice/Speech activity detection', required=False)
    parser.add_argument('--in-emb-dir',
                        help='input directory containing embeddings', required=False)
    parser.add_argument('--out-emb-dir',
                        help='output directory for storing embeddings', required=False)
    parser.add_argument('--norm-list',
                        help='list of normalization files without suffix', required=False)
    parser.add_argument('--in-rttm-dir',
                        help='input directory with rttm files', required=False)
    parser.add_argument('--out-rttm-dir',
                        help='output directory for storing rttm files', required=False)
    parser.add_argument('--out-clusters-dir', required=False,
                        help='output directory for storing clusters - only used when mode is `sre`')
    parser.add_argument('-wav-suffix',
                        help='wav file suffix', required=False, default='.wav')
    parser.add_argument('-vad-suffix',
                        help='Voice Activity Detector file suffix', required=False, default='.lab.gz')
    parser.add_argument('-rttm-suffix',
                        help='rttm file suffix', required=False, default='.rttm')
    parser.add_argument('--min-window-size', default=250,
                        help='minimal window size for embedding clustering in ms', type=int, required=False)
    parser.add_argument('--max-window-size', default=750,
                        help='maximal window size for extracting embedding in ms', type=int, required=False)
    parser.add_argument('--window-overlap',
                        help='overlap in window in ms', type=int, required=False, default=750)
    parser.add_argument('--vad-tolerance', default=0,
                        help='tolerance critetion for ignoring frames of silence', type=float, required=False)
    parser.add_argument('-j', '--num-threads',
                        help='number of processor threads to use', required=False, type=int, default=1)
    parser.add_argument('--max-num-speakers',
                        help='maximal number of speakers', required=False, default=10)

    args = parser.parse_args()

    logger.info(f'Running `{" ".join(sys.argv)}`.')

    set_mkl(1)

    # initialize extractor
    config = Utils.read_config(args.configuration)

    config_mfcc = config['MFCC']
    config_path = os.path.abspath(config_mfcc['config_path'])
    if not os.path.isfile(config_path):
        raise ValueError(f'Path to MFCC configuration `{config_path}` not found.')
    features_extractor = KaldiMFCCFeatureExtraction(
        config_path=config_path, apply_cmvn_sliding=config_mfcc['apply_cmvn_sliding'],
        norm_vars=config_mfcc['norm_vars'], center=config_mfcc['center'], cmn_window=config_mfcc['cmn_window'])

    config_embedding_extractor = config['EmbeddingExtractor']
    embedding_extractor = ONNXXVectorExtraction(onnx_path=os.path.abspath(config_embedding_extractor['onnx_path']))

    config_transforms = config['Transforms']
    mean = config_transforms.get('mean')
    lda = config_transforms.get('lda')
    if lda is not None:
        lda = np.load(lda)
    use_l2_norm = config_transforms.get('use_l2_norm')

    files = [line.rstrip('\n') for line in open(args.input_list)]

    # extract embeddings
    if args.in_emb_dir is None:
        if args.audio_dir is None:
            raise ValueError('At least one of `--in-emb-dir` or `--audio-dir` must be specified.')
        if args.vad_dir is None:
            raise ValueError('`--audio-dir` was specified, `--vad-dir` must be specified too.')
        # process_files(
            # fns=files, wav_dir=args.audio_dir, vad_dir=args.vad_dir, out_dir=args.out_emb_dir,
            # features_extractor=features_extractor, embedding_extractor=embedding_extractor,
            # min_size=args.min_window_size, max_size=args.max_window_size, overlap=args.window_overlap,
            # tolerance=args.vad_tolerance, wav_suffix=args.wav_suffix, vad_suffix=args.vad_suffix,
            # n_jobs=args.num_threads)
        if args.out_emb_dir:
            embeddings = args.out_emb_dir
    else:
        embeddings = args.in_emb_dir

    # initialize normalization
    if args.norm_list is not None:
        norm = Normalization(norm_list=args.norm_list, audio_dir=args.audio_dir,
                             in_rttm_dir=args.in_rttm_dir, in_emb_dir=args.in_emb_dir,
                             out_emb_dir=args.out_emb_dir, min_length=args.min_window_size,
                             embedding_extractor=embedding_extractor, features_extractor=features_extractor,
                             wav_suffix=args.wav_suffix, rttm_suffix=args.rttm_suffix, n_jobs=args.num_threads)
    else:
        norm = None

    # load transformations if specified
    if not norm:
        if mean:
            mean = np.load(mean)
    else:
        mean = norm.mean

    # run diarization
    diar = Diarization(args.input_list, embeddings, embeddings_mean=mean, lda=lda,
                       use_l2_norm=use_l2_norm, norm=norm)
    result = diar.score_embeddings(args.min_window_size, args.max_num_speakers, args.mode)

    if args.mode == 'diarization':
        if args.in_rttm_dir:
            diar.evaluate(scores=result, in_rttm_dir=args.in_rttm_dir, collar_size=0.25, evaluate_overlaps=False)

        if args.out_rttm_dir is not None:
            diar.dump_rttm(result, args.out_rttm_dir)
    else:
        if args.out_clusters_dir:
            for name in result:
                mkdir_p(os.path.join(args.out_clusters_dir, os.path.dirname(name)))
                np.save(os.path.join(args.out_clusters_dir, name), result[name])
