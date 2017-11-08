#! /usr/bin/env python

import argparse
import ctypes
import multiprocessing
import os

from scipy import signal
from scipy.io.wavfile import read

from vbdiar.features.raw2ivec import get_vad, get_segments, get_num_segments, RATE
from vbdiar.scoring.normalization import Normalization
from vbdiar.scoring.diarization import Diarization
from vbdiar.scoring.plda import PLDA
from vbdiar.utils.utils import loginfo, logwarning, Utils
from vbdiar.features.features import Features
from vbdiar.ivectors.fea2ivec import Fea2Ivec
from vbdiar.ivectors.ivec import IvecSet


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


def process_files(fns, wav_dir, vad_dir, out_dir, fea2ivec_obj, min_size, max_size,
                  tolerance, wav_suffix='.wav', vad_suffix='.lab.gz', n_jobs=1):
    """ Process all files from list.

    Args:
        fns (list): name of files to process
        wav_dir (str): directory with wav files
        vad_dir (str): directory with vad files
        out_dir (str): output directory
        fea2ivec_obj (Fea2Ivec): input models for i-vector extraction
        min_size (int): minimal size of window in ms
        max_size (int): maximal size of window in ms
        tolerance (int): accept given number of frames as speech even when it is marked as silence
        wav_suffix (str): suffix of wav files
        vad_suffix (str): suffix of vad files
        n_jobs (int): number of jobs to run in parallel

    Returns:

    """
    kwargs = dict(wav_dir=wav_dir, vad_dir=vad_dir, out_dir=out_dir, fea2ivec_obj=fea2ivec_obj, min_size=min_size,
                  max_size=max_size, wav_suffix=wav_suffix, vad_suffix=vad_suffix, tolerance=tolerance)
    if n_jobs == 1:
        ret = _process_files((fns, kwargs))
    else:
        pool = multiprocessing.Pool(n_jobs)
        ret = pool.map(_process_files, ((part, kwargs) for part in Utils.partition(fns, n_jobs)))
    return ret


def process_file(wav_dir, vad_dir, out_dir, file_name, fea2ivec_obj, min_size, max_size,
                 tolerance, wav_suffix='wav', vad_suffix='lab.gz'):
    """ Process single audio file.

    Args:
        wav_dir (str): directory with wav files
        vad_dir (str): directory with vad files
        out_dir (str): output directory
        file_name (str): name of the file
        fea2ivec_obj (Fea2Ivec): input models for i-vector extraction
        min_size (int): minimal size of window in ms
        max_size (int): maximal size of window in ms
        tolerance (int): accept given number of frames as speech even when it is marked as silence
        wav_suffix (str): suffix of wav files
        vad_suffix (str): suffix of vad files

    """
    loginfo('Processing file {} ...'.format(file_name.split()[0]))
    num_speakers = None
    if len(file_name.split()) > 1:  # number of speakers is defined
        file_name, num_speakers = file_name.split()[0], int(file_name.split()[1])
    wav = '{}.{}'.format(os.path.join(wav_dir, file_name), wav_suffix)
    rate, sig = read(wav)
    if len(sig.shape) != 1:
        raise ValueError('Expected mono as input audio.')
    if rate != RATE:
        logwarning('The input file is expected to be in 8000 Hz, got {} Hz instead, resampling.'.format(rate))
        sig = signal.resample(sig, RATE)

    fea_extractor = Features()
    fea = fea_extractor(sig)
    vad, n_regions, n_frames = get_vad('{}.{}'.format(os.path.join(vad_dir, file_name), vad_suffix), len(fea))

    ivec_set = IvecSet()
    ivec_set.name = file_name
    ivec_set.num_speakers = num_speakers
    for seg in get_segments(vad, min_size, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        if seg[0] > fea.shape[0] - 1 or seg[1] > fea.shape[0] - 1:
            raise ValueError('Unexpected features dimensionality - check VAD input or audio.')
        w = fea2ivec_obj.get_ivec(fea[seg[0]:seg[1]])
        ivec_set.add(w, start, end, mfccs=fea)
    if out_dir is not None:
        Utils.mkdir_p(os.path.join(out_dir, os.path.dirname(file_name)))
        ivec_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))
    else:
        return ivec_set


def set_mkl(num_cores=1):
    """ Set number of cores for mkl library.

        :param num_cores: number of cores
        :type num_cores: int
    """
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_rt.mkl_set_dynamic(ctypes.byref(ctypes.c_int(0)))
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_cores)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract i-vectors used for diarization from audio wav files.')

    # required
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('-c', '--configuration', help='input configuration of models',
                        action='store', dest='configuration', type=str, required=True)

    # not required
    parser.add_argument('--audio-dir', help='directory with audio files in .wav format - 8000Hz, 16bit-s, 1c',
                        action='store', dest='audio_dir', type=str, required=False)
    parser.add_argument('--vad-dir', help='directory with lab files - Voice/Speech activity detection',
                        action='store', dest='vad_dir', type=str, required=False)
    parser.add_argument('--in-ivec-dir', help='input directory containing i-vectors',
                        action='store', dest='in_ivec_dir', type=str, required=False)
    parser.add_argument('--out-ivec-dir', help='output directory for storing i-vectors',
                        action='store', dest='out_ivec_dir', type=str, required=False)
    parser.add_argument('--norm-list', help='list of normalization files without suffix',
                        action='store', dest='norm_list', type=str, required=False)
    parser.add_argument('--in-rttm-dir', help='input directory with rttm files',
                        action='store', dest='in_rttm_dir', type=str, required=False)
    parser.add_argument('--out-rttm-dir', help='output directory for storing rttm files',
                        action='store', dest='out_rttm_dir', type=str, required=False)
    parser.add_argument('-wav-suffix', help='wav file suffix',
                        action='store', dest='wav_suffix', type=str, required=False)
    parser.add_argument('-vad-suffix', help='Voice Activity Detector file suffix',
                        action='store', dest='vad_suffix', type=str, required=False)
    parser.add_argument('-rttm-suffix', help='rttm file suffix',
                        action='store', dest='rttm_suffix', type=str, required=False)
    parser.add_argument('--min-window-size', help='minimal window size for extracting i-vector in ms',
                        action='store', dest='min_window_size', type=int, required=False)
    parser.add_argument('--max-window-size', help='maximal window size for extracting i-vector in ms',
                        action='store', dest='max_window_size', type=int, required=False)
    parser.add_argument('--vad-tolerance', help='tolerance critetion for ignoring frames of silence',
                        action='store', dest='vad_tolerance', type=int, required=False)
    parser.add_argument('-j', '--num-threads', help='number of processor threads to use',
                        action='store', dest='num_threads', type=int, required=False)
    parser.add_argument('--max-num-speakers', help='maximal number of speakers',
                        action='store', dest='max_num_speakers', type=int, required=False)
    parser.set_defaults(num_cores=1)
    parser.set_defaults(max_num_speakers=10)
    parser.set_defaults(wav_suffix='wav')
    parser.set_defaults(vad_suffix='lab.gz')
    parser.set_defaults(rttm_suffix='rttm')
    parser.set_defaults(min_window_size=1000)
    parser.set_defaults(max_window_size=4000)
    parser.set_defaults(vad_tolerance=2)
    args = parser.parse_args()

    set_mkl(1)

    # initialize extractor
    config = Utils.read_config(args.configuration)
    fea2ivec = Fea2Ivec(config['GMM']['model_path'], config['Extractor']['model_path'])
    files = [line.rstrip('\n') for line in open(args.input_list)]

    # extract i-vectors
    if args.in_ivec_dir is None:
        ivec = process_files(
            files, args.audio_dir, args.vad_dir, args.in_ivec_dir, fea2ivec, args.min_window_size,
            args.max_window_size, args.vad_tolerance, args.wav_suffix, args.vad_suffix, args.num_threads)
        if args.out_ivec_dir:
            for ivecset in ivec:
                ivecset.save('{}.{}'.format(os.path.join(args.out_ivec_dir, ivecset.name), 'pkl'))
    else:
        ivec = args.in_ivec_dir

    # initialize PLDA model
    try:
        plda = PLDA(config['PLDA']['model_path'])
    except IOError:
        logwarning('PLDA model initialization failed. Cosine distance will be used instead.')
        plda = None

    # initialize normalization
    if args.norm_list is not None:
        norm = Normalization(args.norm_list, args.audio_dir, args.in_rttm_dir, args.in_ivec_dir, args.out_ivec_dir,
                             fea2ivec, plda, args.wav_suffix, args.rttm_suffix)
    else:
        norm = None

    # run diarization
    diar = Diarization(args.input_list, ivec, norm, plda)
    scores = diar.score_ivec(args.max_num_speakers)
    if args.out_rttm_dir is not None:
        diar.dump_rttm(scores, args.out_rttm_dir)
