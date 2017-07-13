#! /usr/bin/env python

import math
import argparse
import multiprocessing

from scipy.io.wavfile import read
from joblib import Parallel, delayed

from lib.raw2ivec import *
from lib.ivec import IvecSet
from lib.tools import loginfo

from lib.user_exception import GeneralException

from wav2ivec import init, get_ivec, get_vad, get_mfccs


def process_file(wav_dir, vad_dir, out_dir, file_name, model, min_size, max_size,
                 tolerance, wav_suffix='.wav', vad_suffix='.lab.gz'):
    loginfo('[wav2ivec.process_file] Processing file {} ...'.format(file_name))
    ubm_weights, ubm_means, ubm_covs, ubm_norm, gmm_model, numg, dimf, v, mvvt = model
    wav = os.path.join(wav_dir, file_name) + wav_suffix
    rate, sig = read(wav)
    if rate != 8000:
        raise GeneralException(
            '[wav2ivec.process_file] The input file is expected to be in 8000 Hz, got {} instead.'.format(rate)
        )
    if ADDDITHER > 0.0:
        loginfo('[wav2ivec.process_file] Adding dither ...')
        sig = features.add_dither(sig, ADDDITHER)

    fea = get_mfccs(sig)
    vad, n_regions, n_frames = get_vad(vad_dir, file_name, vad_suffix, sig, fea)

    ivec_set = IvecSet()
    for seg in get_segments(vad, min_size, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        w = get_ivec(fea[seg[0]:seg[1] + 1], numg, dimf, gmm_model, ubm_means, ubm_norm, v, mvvt)
        ivec_set.add(w, start, end)

    ivec_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))


def get_segments(vad, min_size, max_size, tolerance):
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
    size = segment[1] - segment[0]
    num_segments = int(math.ceil(size / max_size))
    size_segment = size / num_segments
    for ii in range(num_segments):
        yield (segment[0] + ii * size_segment, segment[0] + (ii + 1) * size_segment)


def get_num_frames(n):
    return 1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000)


def get_num_segments(n):
    return int(n * (TARGETRATE / 10000) - (TARGETRATE / 10000) + (WINDOWSIZE / 10000))


def get_clusters(vad, min_size, tolerance=10):
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
    parser.add_argument('--min-collar-size', help='minimal collar size for extracting i-vector in ms',
                        action='store', dest='min_collar_size', type=int, required=False)
    parser.add_argument('--max-collar-size', help='maximal collar size for extracting i-vector in ms',
                        action='store', dest='max_collar_size', type=int, required=False)
    parser.add_argument('--vad-tolerance', help='tolerance critetion for ignoring frames of silence',
                        action='store', dest='vad_tolerance', type=int, required=False)
    parser.add_argument('-j', '--num-cores', help='number of processor cores to use',
                        action='store', dest='num_cores', type=int, required=False)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    parser.set_defaults(wav_suffix='.wav')
    parser.set_defaults(vad_suffix='.lab.gz')
    parser.set_defaults(min_collar_size=1000)
    parser.set_defaults(max_collar_size=2000)
    parser.set_defaults(vad_tolerance=10)
    args = parser.parse_args()

    models = init(args.ubm_file, args.v_file)
    loginfo('[wav2ivec.main] Using {} processor cores ...'.format(args.num_cores))
    files = [line.rstrip('\n') for line in open(args.input_list)]
    Parallel(n_jobs=args.num_cores)(delayed(process_file)(
        args.audio_dir, args.vad_dir, args.out_dir, f, models, args.min_collar_size, args.max_collar_size,
        args.vad_tolerance, wav_suffix=args.wav_suffix, vad_suffix=args.vad_suffix)
        for f in files)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
