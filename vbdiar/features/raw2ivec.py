#!/usr/bin/env python

import errno
import os
import sys

import numpy as np
import scipy.io.wavfile as spiowav


################################################################################
################################################################################
from vbdiar.utils.utils import loginfo

RATE = 8000
SOURCERATE = 1250
TARGETRATE = 100000

LOFREQ = 120
HIFREQ = 3800

ZMEANSOURCE = True
WINDOWSIZE = 250000.0
USEHAMMING = True
PREEMCOEF = 0.97
NUMCHANS = 24
CEPLIFTER = 22
NUMCEPS = 19
ADDDITHER = 1.0
RAWENERGY = True
ENORMALISE = True

deltawindow = accwindow = 2

cmvn_lc = 150
cmvn_rc = 150

fs = 1e7 / SOURCERATE


################################################################################
################################################################################

class NoVadException(Exception):
    """ No VAD exception - raised when there is no VAD definition for a file
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def row(v):
    return v.reshape((1, v.size))


def mkdir_p(path):
    """ mkdir 
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_ubm(fname):
    """ This function will load the UBM from the file and will return the
        parameters in three separate variables
    """
    gmm = np.load(fname)

    n_superdims = (gmm.shape[1] - 1) / 2

    weights = gmm[:, 0]
    means = gmm[:, 1:(n_superdims + 1)]
    covs = gmm[:, (n_superdims + 1):]

    return weights, means, covs


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
    num_segments = int(np.math.ceil(size / max_size))
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


def load_vad_lab_as_bool_vec(lab_file):
    lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

    if lab_cont.shape[1] == 0:
        return np.empty(0), 0, 0

    # else:
    #     lab_cont = lab_cont.reshape((-1,lab_cont.shape[0]))

    if lab_cont.shape[1] == 3:
        lab_cont = lab_cont[lab_cont[:, 2] == 'sp', :][:, [0, 1]]

    n_regions = lab_cont.shape[0]
    ii = 0
    while True:
        try:
            start1, end1 = float(lab_cont[ii][0]), float(lab_cont[ii][1])
            jj = ii + 1
            start2, end2 = float(lab_cont[jj][0]), float(lab_cont[jj][1])
            if end1 >= start2:
                lab_cont = np.delete(lab_cont, ii, axis=0)
                ii -= 1
                lab_cont[jj - 1][0] = str(start1)
                lab_cont[jj - 1][1] = str(max(end1, end2))
            ii += 1
        except IndexError:
            break

    vad = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
    vad[1] += 1  # Paja's bug!!!

    if not vad.size:
        return np.empty(0, dtype=bool)

    npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
    npc2 = np.c_[vad[0] - np.r_[0, vad[1, :-1]], vad[1] - vad[0]]

    out = np.repeat(npc1, npc2.flat)

    n_frames = sum(out)

    return out, n_regions, n_frames


def compute_vad(s, win_length=160, win_overlap=80):
    v = evad.compute_vad(s, win_length=win_length, win_overlap=win_overlap, n_realignment=10)

    n_frames = sum(v)
    n_regions = n_frames

    return v, n_regions, n_frames


def split_seq(seq, size):
    """ Split up seq in pieces of size """
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def normalize_stats(n, f, ubm_means, ubm_norm):
    """ Center the first-order UBM stats around UBM means and normalize 
        by the UBM covariance 
    """
    n_gauss = n.shape[0]
    n_superdim = f.shape[0]
    n_fdim = n_superdim / n_gauss

    f0 = f - ubm_means * np.kron(np.ones((n_fdim, 1), dtype=n.dtype), n).transpose()
    f0 = f0 * ubm_norm

    return n, f0


################################################################################
################################################################################
def main(argv):
    fbank_mx = features.mel_fbank_mx(winlen_nfft=WINDOWSIZE / SOURCERATE,
                                     fs=fs,
                                     NUMCHANS=NUMCHANS,
                                     LOFREQ=LOFREQ,
                                     HIFREQ=HIFREQ)

    scp_list = sys.argv[1]
    vad_dir = sys.argv[2]
    wav_dir = sys.argv[3]
    ubm_file = sys.argv[4]
    v_file = sys.argv[5]
    out_dir = sys.argv[6]

    print 'Loading UBM from', ubm_file
    ubm_weights, ubm_means, ubm_covs = load_ubm(ubm_file)
    GMM = gmm.gmm_eval_prep(ubm_weights, ubm_means, ubm_covs)

    numG = ubm_means.shape[0]
    dimF = ubm_means.shape[1]

    # normalization of statistics - precomputing matrices
    if ubm_covs.shape[1] == dimF:
        ubm_norm = 1 / np.sqrt(ubm_covs);

    print 'Loading T matrix from ', v_file, '...'
    v = np.loadtxt(v_file, dtype=np.float32)

    print 'Computing MVVT ...'
    MVVT = iv.compute_VtV(v, numG)

    print 'Loading list of files to process from ' + scp_list
    seg_list = np.atleast_1d(np.loadtxt(scp_list, dtype=object))

    # extract all sub-dir names
    for dir in set(map(os.path.dirname, seg_list)):
        mkdir_p(out_dir + '/' + dir)

    # go over the scp and process the audio files
    for ii, fn in enumerate(seg_list, 1):
        try:
            print 'Processing ', ii, '/', len(seg_list), fn
            np.random.seed(777)

            wav_file = wav_dir + '/' + fn + '.wav'
            raw_file = wav_dir + '/' + fn + '.raw'
            lab_file = vad_dir + '/' + fn + '.lab.gz'
            ivec_out_file = out_dir + '/' + fn + '.ivec'

            if os.path.isfile(wav_file):
                print '  Reading wave file from ' + wav_file,
                rate, sig = spiowav.read(wav_file)

                if rate != 8000:
                    raise Exception(
                        'The input file ' + wav_file + ' is expected to be in 8000 Hz sampling rate, but ' + repr(
                            rate) + ' Hz detected')

            else:
                print '  Reading raw 8000Hz, 16bit-s, 1c,  file from ' + raw_file,
                sig = np.fromfile(raw_file, dtype='int16')

            print '[t=' + repr(len(sig) / fs) + ' seconds, fs=' + repr(fs) + 'Hz, n=' + repr(len(sig)) + ' samples]'

            if ADDDITHER > 0.0:
                print '  Adding dither'
                sig = features.add_dither(sig, ADDDITHER)

            print '  Extracting features',
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

            print '[n=' + repr(len(fea)) + ' frames]'

            print '  Adding derivatives'
            # [add_deriv] step
            fea = features.add_deriv(fea, (deltawindow, accwindow))

            print '  Reshaping to SFeaCat convention'
            # [reshape] step
            fea = fea.reshape(fea.shape[0], 3, -1).transpose((0, 2, 1)).reshape(fea.shape[0],
                                                                                -1)  # re-order coeffs like SFeaCut

            if vad_dir == "auto":
                print '  Computing VAD '
                vad, n_regions, n_frames = compute_vad(sig, win_length=WINDOWSIZE / SOURCERATE,
                                                       win_overlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE)[:len(fea)]
            else:
                print '  Loading VAD definition from ' + lab_file
                vad, n_regions, n_frames = load_vad_lab_as_bool_vec(lab_file)[:len(fea)]

            print '  Applying VAD [#frames=' + repr(n_frames) + ', #regions=' + repr(n_regions) + ']'
            fea = fea[vad, ...]

            if len(fea) < 3:
                raise NoVadException('Too few frames left: ' + str(len(fea)))

            print '  Applying floating CMVN'
            fea = features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)

            n_data, d_data = fea.shape

            l = 0;
            lc = 0
            n = np.zeros((numG), dtype=np.float32)
            f = np.zeros((numG, dimF), dtype=np.float32)

            print '  Computing stats ...',
            # Note that we compute the stats in in sub-chunks due to memory optimization
            #
            seq_data = split_seq(range(n_data), 1000)
            for i in range(len(seq_data)):
                dd = fea[seq_data[i], :]
                l1, n1, f1 = gmm.gmm_eval(dd, GMM, return_accums=1)
                l = l + l1.sum()
                lc = lc + l1.shape[0]
                n = n + n1;
                f = f + f1;

            print '[avg llh=' + repr(l / lc) + ', #frames=' + repr(n_data) + ']'

            n, f = normalize_stats(n, f, ubm_means, ubm_norm)

            f = row(f.astype(v.dtype))
            n = row(n.astype(v.dtype))

            print '  Computing i-vector'
            w = iv.estimate_i(n, f, v, MVVT).T

            # write it to the disk
            print '  Saving ivec to:', ivec_out_file
            # np.savetxt(ivec_out_file, w.ravel(), newline=' ', fmt='%f')
            ivio.write_binary_ivector(ivec_out_file, w.ravel(), n_data / 100.0)

        except NoVadException as e:
            print e
            print "Warning: No features generated for segment: " + fn

        except:
            raise


if __name__ == '__main__':
    main(sys.argv[1:])
