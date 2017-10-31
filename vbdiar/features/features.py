#!/usr/bin/env python

import numpy as np
import scipy.fftpack


class Features(object):
    def __init__(self):
        self.window = np.hamming(200)
        self.noverlap = 120
        self.deltawindow = self.accwindow = 2
        self.cmvn_lc = self.cmvn_rc = 150
        self.fbank_mx = Features.mel_fbank_mx(self.window.size, fs=8000, NUMCHANS=24)
        np.seterr(divide='ignore', invalid='ignore')

    def __str__(self):
        return 'MFCC 19E DG'

    def __call__(self, x):
        fea = Features.add_dither(x, 1.0)
        fea = Features.mfcc_htk(fea, self.window, self.noverlap, self.fbank_mx, NUMCEPS=19, USEPOWER=False,
                                ZMEANSOURCE=True, PREEMCOEF=0.97, RAWENERGY=True, USEHAMMING=True,
                                ENORMALISE=True, _0=None, _E='last')
        return fea

    @staticmethod
    def add_dither(x, level=8):
        return x + level * (np.random.rand(*x.shape)*2-1)

    @staticmethod
    def mel_inv(x):
        return (np.exp(x/1127.)-1.)*700.

    @staticmethod
    def mel(x):
        return 1127.*np.log(1. + x/700.)

    @staticmethod
    def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None):
        """ Returns mel filterbank as an array (NFFT/2+1 x NUMCHANS)
        winlen_nfft - Typically the window length as used in mfcc_htk() call. It is
                      used to determine number of samples for FFT computation (NFFT).
                      If positive, the value (window lenght) is rounded up to the
                      next higher power of two to obtain HTK-compatible NFFT.
                      If negative, NFFT is set to -winlen_nfft. In such case, the
                      parameter nfft in mfcc_htk() call should be set likewise.
        fs          - sampling frequency (Hz, i.e. 1e7/SOURCERATE)
        NUMCHANS    - number of filter bank bands
        LOFREQ      - frequency (Hz) where the first filter strats
        HIFREQ      - frequency (Hz) where the last  filter ends (default fs/2)
        warp_fn     - function for frequency warping and its inverse
        inv_warp_fn - inverse function to warp_fn
        """

        if not HIFREQ:
            HIFREQ = 0.5 * fs
        nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

        fbin_mel = Features.mel(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
        cbin_mel = np.linspace(Features.mel(LOFREQ), Features.mel(HIFREQ), NUMCHANS + 2)
        cind = np.floor(Features.mel_inv(cbin_mel) / fs * nfft).astype(int) + 1
        mfb = np.zeros((len(fbin_mel), NUMCHANS))
        for i in xrange(NUMCHANS):
            mfb[cind[i]  :cind[i+1], i] = (cbin_mel[i]  -fbin_mel[cind[i]  :cind[i+1]]) / (cbin_mel[i]  -cbin_mel[i+1])
            mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2]-fbin_mel[cind[i+1]:cind[i+2]]) / (cbin_mel[i+2]-cbin_mel[i+1])
        if LOFREQ > 0.0 and float(LOFREQ)/fs*nfft+0.5 > cind[0]: mfb[cind[0],:] = 0.0 # Just to be HTK compatible
        return mfb

    @staticmethod
    def mfcc_htk(x, window, noverlap, fbank_mx, nfft=None,
                 _0="last", _E=None, NUMCEPS=12,
                 USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, CEPLIFTER=22.0, ZMEANSOURCE=False,
                 ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
        """MFCC Mel Frequency Cepstral Coefficients
        Returns NUMCEPS-by-M matrix of MFCC coeficients extracted form signal x,
        where M is the number of extracted frames, which can be computed as
        floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
        have the following meaning:
        x         - input signal
        window    - frame window lentgth (in samples, i.e. WINDOWSIZE/SOURCERATE)
                    or vector of widow weights override default windowing function
                    (see option USEHAMMING)
        noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
        fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                    Note that this must be compatible with the parameter 'nfft'.
        nfft      - number of samples for FFT computation. By default, it is  set in the
                    HTK-compatible way to the window length rounded up to the next higher
                    pover of two.
        _0, _E    - include C0 or/and energy as the "first" or the "last" coefficient(s)
                    of each feature vector. The possible values are: "first", "last", None.
                    If both C0 and energy are used, energy will be the very first or the
                    very last coefficient.
        Remaining options have exactly the same meaning as in HTK.
        See also:
          mel_fbank_mx:
              to obtain the matrix for the parameter fbank_mx
          add_deriv:
              for adding delta, double delta, ... coefficients
          add_dither:
              for adding dithering in HTK-like fashion
        """

        dct_mx = Features.dct_basis(NUMCEPS + 1, fbank_mx.shape[1]).T
        dct_mx[:, 0] = np.sqrt(2.0 / fbank_mx.shape[1])
        if type(USEPOWER) == bool:
            USEPOWER += 1
        if np.isscalar(window):
            window = np.hamming(window) if USEHAMMING else np.ones(window)
        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(window.size)))
        x = Features.framing(x.astype("float"), window.size, window.size - noverlap).copy()
        if ZMEANSOURCE:
            x -= x.mean(axis=1)[:, np.newaxis]
        if _E is not None and RAWENERGY:
            energy = np.log((x ** 2).sum(axis=1))
        if PREEMCOEF is not None:
            x = Features.preemphasis(x, PREEMCOEF)
        x *= window
        if _E is not None and not RAWENERGY:
            energy = np.log((x ** 2).sum(axis=1))
        # x = np.abs(scipy.fftpack.fft(x, nfft))
        # x = x[:,:x.shape[1]/2+1]
        x = np.abs(np.fft.rfft(x, nfft))
        x = np.log(np.maximum(1.0, (x ** USEPOWER).dot(fbank_mx))).dot(dct_mx)
        if CEPLIFTER is not None and CEPLIFTER > 0:
            x *= 1.0 + 0.5 * CEPLIFTER * np.sin(np.pi * np.arange(NUMCEPS + 1) / CEPLIFTER)
        if _E is not None and ENORMALISE:
            energy = (energy - energy.max()) * ESCALE + 1.0
            min_val = -np.log(10 ** (SILFLOOR / 10.)) * ESCALE + 1.0
            energy[energy < min_val] = min_val

        return np.hstack(([energy[:, np.newaxis]] if _E == "first" else []) +
                         ([x[:, :1]] if _0 == "first" else []) +
                         [x[:, 1:]] +
                         ([x[:, :1]] if (_0 in ["last", True])  else []) +
                         ([energy[:, np.newaxis]] if (_E in ["last", True])  else []))

    @staticmethod
    def framing(a, window, shift=1):
        shape = (int(round((a.shape[0] - window) / shift + 1)), int(window)) + a.shape[1:]
        strides = (int(a.strides[0] * shift), a.strides[0]) + a.strides[1:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def dct_basis(nbasis, length):
        # the same DCT as in matlab
        return scipy.fftpack.idct(np.eye(nbasis, length), norm='ortho')

    @staticmethod
    def preemphasis(x, coef=0.97):
        return x - np.c_[x[..., :1], x[..., :-1]] * coef
