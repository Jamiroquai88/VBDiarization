#! /usr/bin/env python

import os
import sys
import argparse
import multiprocessing

from scipy.io.wavfile import read
from joblib import Parallel, delayed

from lib.tools import loginfo
from lib.raw2ivec import *

from lib.user_exception import GeneralException


def init(ubm_file, v_file):
    loginfo('[wav2ivec.init] Loading UBM file ...')
    ubm_weights, ubm_means, ubm_covs = load_ubm(ubm_file)
    gmm_model = gmm.gmm_eval_prep(ubm_weights, ubm_means, ubm_covs)
    numg = ubm_means.shape[0]
    dimf = ubm_means.shape[1]
    if ubm_covs.shape[1] == dimf:
        ubm_norm = 1 / np.sqrt(ubm_covs)
    else:
        ubm_norm = None
    loginfo('[wav2ivec.init] Loading V model file ...')
    v = np.load(v_file)
    mvvt = iv.compute_VtV(v, numg)
    return ubm_weights, ubm_means, ubm_covs, ubm_norm, gmm_model, numg, dimf, v, mvvt


def get_mfccs(sig):
    loginfo('[wav2ivec.get_mfccs] Extracting MFCC features ...')
    fbank_mx = features.mel_fbank_mx(winlen_nfft=WINDOWSIZE / SOURCERATE,
                                     fs=fs,
                                     NUMCHANS=NUMCHANS,
                                     LOFREQ=LOFREQ,
                                     HIFREQ=HIFREQ)
    fea = features.mfcc_htk(sig,
                            window=WINDOWSIZE / SOURCERATE,
                            noverlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE,
                            fbank_mx=fbank_mx,
                            _0='first',
                            NUMCEPS=NUMCEPS,
                            RAWENERGY=RAWENERGY,
                            PREEMCOEF=PREEMCOEF,
                            CEPLIFTER=CEPLIFTER,
                            ZMEANSOURCE=ZMEANSOURCE,
                            ENORMALISE=ENORMALISE,
                            ESCALE=0.1,
                            SILFLOOR=50.0,
                            USEHAMMING=True)

    loginfo('[wav2ivec.get_mfccs] Adding derivatives ...')
    fea = features.add_deriv(fea, (deltawindow, accwindow))

    loginfo('[wav2ivec.get_mfccs] Reshaping to SFeaCat conventions ...')
    return fea.reshape(fea.shape[0], 3, -1).transpose((0, 2, 1)).reshape(fea.shape[0], -1)


def get_vad(vad_dir, file_name, vad_suffix, sig, fea):
    if vad_dir is None:
        loginfo('[wav2ivec.get_vad] Computing VAD ...')
        return compute_vad(sig, win_length=WINDOWSIZE / SOURCERATE,
                           win_overlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE)[:len(fea)]
    else:
        vad = os.path.join(vad_dir, file_name) + vad_suffix
        loginfo('[wav2ivec.get_vad] Loading VAD from file {} ...'.format(vad))
        return load_vad_lab_as_bool_vec(vad)[:len(fea)]


def get_ivec(fea, numg, dimf, gmm_model, ubm_means, ubm_norm, v, mvvt):
    loginfo('[wav2ivec.get_ivec] Applying floating CMVN ...')
    fea = features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)
    n_data, d_data = fea.shape
    l = 0
    lc = 0
    n = np.zeros(numg, dtype=np.float32)
    f = np.zeros((numg, dimf), dtype=np.float32)

    loginfo('[wav2ivec.get_ivec] Computing statistics ...')
    seq_data = split_seq(range(n_data), 1000)
    for i in range(len(seq_data)):
        dd = fea[seq_data[i], :]
        l1, n1, f1 = gmm.gmm_eval(dd, gmm_model, return_accums=1)
        l = l + l1.sum()
        lc = lc + l1.shape[0]
        n = n + n1
        f = f + f1
    n, f = normalize_stats(n, f, ubm_means, ubm_norm)
    f = row(f.astype(v.dtype))
    n = row(n.astype(v.dtype))

    loginfo('[wav2ivec.get_ivec] Computing i-vector ...')
    return iv.estimate_i(n, f, v, mvvt).T


def process_file(wav_dir, vad_dir, out_dir, file_name, model, wav_suffix='.wav', vad_suffix='.lab.gz'):
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

    fea = fea[vad, ...]
    w = get_ivec(fea, numg, dimf, gmm_model, ubm_means, ubm_norm, v, mvvt)
    np.save(os.path.join(out_dir, file_name), w)


def main(argv):
    parser = argparse.ArgumentParser('Extract i-vector from audio wav files 1:1.')
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
                        action='store', dest='out_dir', type=str, required=False)
    parser.add_argument('--ubm-file', help='Universal Background Model file',
                        action='store', dest='ubm_file', type=str, required=True)
    parser.add_argument('--v-file', help='V Model file',
                        action='store', dest='v_file', type=str, required=True)
    parser.add_argument('-j', '--num-cores', help='number of processor cores to use',
                        action='store', dest='num_cores', type=int, required=False)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    parser.set_defaults(wav_suffix='.wav')
    parser.set_defaults(vad_suffix='.lab.gz')
    args = parser.parse_args()

    models = init(args.ubm_file, args.v_file)
    loginfo('[wav2ivec.main] Using {} processor cores ...'.format(args.num_cores))
    files = [line.rstrip('\n') for line in open(args.input_list)]
    Parallel(n_jobs=args.num_cores)(delayed(process_file)(
        args.audio_dir, args.vad_dir, args.out_dir, f, models, wav_suffix=args.wav_suffix, vad_suffix=args.vad_suffix)
        for f in files)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
