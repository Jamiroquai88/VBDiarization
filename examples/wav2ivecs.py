#! /usr/bin/env python

import argparse
import ctypes
import math
import multiprocessing
import os

from scipy import signal
from scipy.io.wavfile import read

from vbdiar.utils.utils import loginfo, logwarning, Utils
from vbdiar.features.features import Features
from vbdiar.ivectors.fea2ivec import Fea2Ivec
from vbdiar.ivectors.ivec import IvecSet
from vbdiar.vad.vad import load_vad_lab_as_bool_vec

RATE = 8000
SOURCERATE = 1250
TARGETRATE = 100000
WINDOWSIZE = 250000.0


def _process_files(dargs):
    """

    Args:
        dargs:

    Returns:

    """
    fns, kwargs = dargs
    for fn in fns:
        process_file(file_name=fn, **kwargs)


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
        _process_files((fns, kwargs))
    else:
        pool = multiprocessing.Pool(n_jobs)
        pool.map(_process_files, ((part, kwargs) for part in Utils.partition(fns, n_jobs)))


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
    if len(file_name.split()) > 1:  # number of speakers is defined
        file_name = file_name.split()[0]
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
    for seg in get_segments(vad, min_size, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        if seg[0] > fea.shape[0] - 1 or seg[1] > fea.shape[0] - 1:
            raise ValueError('Unexpected features dimensionality - check VAD input or audio.')
        w = fea2ivec_obj.get_ivec(fea[seg[0]:seg[1]])
        ivec_set.add(w, start, end, mfccs=fea)
    Utils.mkdir_p(os.path.join(out_dir, os.path.dirname(file_name)))
    ivec_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))


def get_vad(file_name, fea_len):
    """ Load .lab file as bool vector.

    Args:
        file_name (str): path to .lab file
        fea_len (int): length of features

    Returns:
        np.array: bool vector
    """

    loginfo('Loading VAD from file {} ...'.format(file_name))
    return load_vad_lab_as_bool_vec(file_name)[:fea_len]


def get_segments(vad, min_size, max_size, tolerance):
    """ Return clustered speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered segments
        :rtype: list
    """
    clusters = get_clusters(vad, get_num_frames(min_size), tolerance)
    segments = []
    max_frames = get_num_frames(max_size)
    for item in clusters.values():
        if item[1] - item[0] > max_frames:
            for ss in split_segment(item, max_frames):
                segments.append(ss)
        else:
            segments.append(item)
    return segments


def split_segment(segment, max_size):
    """ Split segment to more with adaptive size.

        :param segment: input segment
        :type segment: tuple
        :param max_size: maximal size of window in ms
        :type max_size: int
        :returns: splitted segment
        :rtype: list
    """
    size = segment[1] - segment[0]
    num_segments = int(math.ceil(size / max_size))
    size_segment = size / num_segments
    for ii in range(num_segments):
        yield (segment[0] + ii * size_segment, segment[0] + (ii + 1) * size_segment)


def get_num_frames(n):
    """ Get number of frames from ms.

        :param n: number of ms
        :type n: int
        :returns: number of frames
        :rtype: int

        >>> get_num_frames(25)
        1
        >>> get_num_frames(35)
        2
    """
    return int(1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000))


def get_num_segments(n):
    """ Get count of ms from number of frames.

        :param n: number of frames
        :type n: int
        :returns: number of ms
        :rtype: int

        >>> get_num_segments(1)
        25
        >>> get_num_segments(2)
        35

    """
    return int(n * (TARGETRATE / 10000) - (TARGETRATE / 10000) + (WINDOWSIZE / 10000))


def get_clusters(vad, min_size, tolerance=10):
    """ Cluster speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered speech segments
        :rtype: dict
    """
    num_prev = 0
    in_tolerance = 0
    num_clusters = 0
    clusters = {}
    for ii, frame in enumerate(vad):
        if frame:
            num_prev += 1
        else:
            in_tolerance += 1
            if in_tolerance > tolerance:
                if num_prev > min_size:
                    clusters[num_clusters] = (ii - num_prev, ii)
                    num_clusters += 1
                num_prev = 0
                in_tolerance = 0
    return clusters


def set_mkl(num_cores=1):
    """ Set number of cores for mkl library.

        :param num_cores: number of cores
        :type num_cores: int
    """
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_cores)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract i-vectors used for diarization from audio wav files.')
    # required
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('-c', '--configuration', help='input configuration of models',
                        action='store', dest='configuration', type=str, required=True)
    parser.add_argument('--audio-dir', help='directory with audio files in .wav format - 8000Hz, 16bit-s, 1c',
                        action='store', dest='audio_dir', type=str, required=True)
    parser.add_argument('--out-dir', help='output directory for storing i-vectors',
                        action='store', dest='out_dir', type=str, required=True)
    parser.add_argument('--vad-dir', help='directory with lab files - Voice/Speech activity detection',
                        action='store', dest='vad_dir', type=str, required=True)

    # not required
    parser.add_argument('-wav-suffix', help='wav file suffix',
                        action='store', dest='wav_suffix', type=str, required=False)
    parser.add_argument('-vad-suffix', help='Voice Activity Detector file suffix',
                        action='store', dest='vad_suffix', type=str, required=False)

    parser.add_argument('--min-window-size', help='minimal window size for extracting i-vector in ms',
                        action='store', dest='min_window_size', type=int, required=False)
    parser.add_argument('--max-window-size', help='maximal window size for extracting i-vector in ms',
                        action='store', dest='max_window_size', type=int, required=False)
    parser.add_argument('--vad-tolerance', help='tolerance critetion for ignoring frames of silence',
                        action='store', dest='vad_tolerance', type=int, required=False)
    parser.add_argument('-j', '--num-threads', help='number of processor threads to use',
                        action='store', dest='num_threads', type=int, required=False)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    parser.set_defaults(wav_suffix='wav')
    parser.set_defaults(vad_suffix='lab.gz')
    parser.set_defaults(min_window_size=1000)
    parser.set_defaults(max_window_size=2000)
    parser.set_defaults(vad_tolerance=5)
    args = parser.parse_args()

    config = Utils.read_config(args.configuration)
    fea2ivec = Fea2Ivec(config['GMM']['model_path'], config['Extractor']['model_path'])
    loginfo('Setting {} processor cores for the MKL library ...'.format(args.num_cores))
    set_mkl(1)
    files = [line.rstrip('\n') for line in open(args.input_list)]
    process_files(files, args.audio_dir, args.vad_dir, args.out_dir, fea2ivec, args.min_window_size,
                  args.max_window_size, args.vad_tolerance, args.wav_suffix, args.vad_suffix, args.num_threads)
