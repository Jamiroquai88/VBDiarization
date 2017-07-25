#! /usr/bin/env python

import math
import argparse
import multiprocessing
from scipy import signal
from scipy.io.wavfile import read

from lib.raw2ivec import *
from lib.ivec import IvecSet
from lib.tools import loginfo, logwarning, Tools


from wav2ivec import init, get_ivec, get_vad, get_mfccs, set_mkl


def process_file(wav_dir, vad_dir, out_dir, file_name, model, min_size, max_size,
                 tolerance, wav_suffix='.wav', vad_suffix='.lab.gz'):
    """ Extract i-vectors from wav file.

        :param wav_dir: directory with wav files
        :type wav_dir: str
        :param vad_dir: directory with vad files
        :type vad_dir: str
        :param out_dir: output directory
        :type out_dir: str
        :param file_name: name of the file
        :type file_name: str
        :param model: input models for i-vector extraction
        :type model: tuple
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :param wav_suffix: suffix of wav files
        :type wav_suffix: str
        :param vad_suffix: suffix of vad files
        :type vad_suffix
    """
    loginfo('[wav2ivecs.process_file] Processing file {} ...'.format(file_name))
    ubm_weights, ubm_means, ubm_covs, ubm_norm, gmm_model, numg, dimf, v, mvvt = model
    wav = os.path.join(wav_dir, file_name) + wav_suffix
    rate, sig = read(wav)
    if rate != 8000:
        logwarning('[wav2ivecs.process_file] '
                   'The input file is expected to be in 8000 Hz, got {} Hz instead, resampling.'.format(rate))
        sig = signal.resample(sig, 8000)
    if ADDDITHER > 0.0:
        loginfo('[wav2ivecs.process_file] Adding dither ...')
        sig = features.add_dither(sig, ADDDITHER)

    fea = get_mfccs(sig)
    vad, n_regions, n_frames = get_vad(vad_dir, file_name, vad_suffix, sig, fea)

    ivec_set = IvecSet()
    ivec_set.name = file_name
    for seg in get_segments(vad, min_size, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        loginfo('[wav2ivecs.process_file] Processing speech segment from {} ms to {} ms ...'.format(start, end))
        w = get_ivec(fea[seg[0]:seg[1] + 1], numg, dimf, gmm_model, ubm_means, ubm_norm, v, mvvt)
        if w is None:
            continue
        ivec_set.add(w, start, end)
    Tools.mkdir_p(os.path.join(out_dir, os.path.dirname(file_name)))
    ivec_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))


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
    """
    return int(1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000))


def get_num_segments(n):
    """ Get count of ms from number of frames.

        :param n: number of frames
        :type n: int
        :returns: number of ms
        :rtype: int
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
        :rtype: list
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


def main(argv):
    parser = argparse.ArgumentParser('Extract i-vectors used for diarization from audio wav files.')
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('--audio-dir', help='directory with audio files in .wav format - 8000Hz, 16bit-s, 1c',
                        action='store', dest='audio_dir', type=str, required=True)
    parser.add_argument('-wav-suffix', help='wav file suffix',
                        action='store', dest='wav_suffix', type=str, required=False)
    parser.add_argument('--vad-dir', help='directory with lab files - Voice/Speech activity detection',
                        action='store', dest='vad_dir', type=str, required=False)
    parser.add_argument('-vad-suffix', help='Voice Activity Detector file suffix',
                        action='store', dest='vad_suffix', type=str, required=False)
    parser.add_argument('--out-dir', help='output directory for storing i-vectors',
                        action='store', dest='out_dir', type=str, required=True)
    parser.add_argument('--ubm-file', help='Universal Background Model file',
                        action='store', dest='ubm_file', type=str, required=True)
    parser.add_argument('--v-file', help='V Model file',
                        action='store', dest='v_file', type=str, required=True)
    parser.add_argument('--min-window-size', help='minimal window size for extracting i-vector in ms',
                        action='store', dest='min_window_size', type=int, required=False)
    parser.add_argument('--max-window-size', help='maximal window size for extracting i-vector in ms',
                        action='store', dest='max_window_size', type=int, required=False)
    parser.add_argument('--vad-tolerance', help='tolerance critetion for ignoring frames of silence',
                        action='store', dest='vad_tolerance', type=int, required=False)
    parser.add_argument('-j', '--num-cores', help='number of processor cores to use',
                        action='store', dest='num_cores', type=int, required=False)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    parser.set_defaults(wav_suffix='.wav')
    parser.set_defaults(vad_suffix='.lab.gz')
    parser.set_defaults(min_window_size=1000)
    parser.set_defaults(max_window_size=2000)
    parser.set_defaults(vad_tolerance=5)
    args = parser.parse_args()

    models = init(args.ubm_file, args.v_file)
    loginfo('[wav2ivecs.main] Setting {} processor cores for the MKL library ...'.format(args.num_cores))
    set_mkl(args.num_cores)
    files = [line.rstrip('\n') for line in open(args.input_list)]
    for f in files:
        process_file(args.audio_dir, args.vad_dir, args.out_dir, f, models, args.min_window_size, args.max_window_size,
                     args.vad_tolerance, wav_suffix=args.wav_suffix, vad_suffix=args.vad_suffix)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
