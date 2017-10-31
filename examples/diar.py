#! /usr/bin/env python

import argparse
import multiprocessing
import os
import sys

from wav2ivecs import set_mkl
from vbdiar.scoring.plda import PLDA
from vbdiar.utils.utils import loginfo, logwarning, Utils
from vbdiar.scoring.diarization import Diarization


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run diarization on input data with PLDA model.')
    # required
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('--ivecs-dir', help='directory containing i-vectos using class IvecSet - pickle format',
                        action='store', dest='ivecs_dir', type=str, required=True)
    parser.add_argument('-c', '--configuration', help='input configuration of models',
                        action='store', dest='configuration', type=str, required=True)

    # not required
    parser.add_argument('--norm-list', help='list of input normalization files without suffix in .npy format',
                        action='store', dest='norm_list', type=str, required=False)
    parser.add_argument('--out-dir', help='output directory for storing .rttm files',
                        action='store', dest='out_dir', type=str, required=False)
    parser.add_argument('--reference', help='reference rttm file for system scoring',
                        action='store', dest='reference', type=str, required=False)
    parser.add_argument('-j', '--num-cores', help='number of processor cores to use',
                        action='store', dest='num_cores', type=int, required=False)
    parser.set_defaults(norm_list=None)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    args = parser.parse_args()

    if args.reference is None and args.out_dir is None:
        parser.print_help()
        sys.stderr.write('At least one of --reference and --out-dir must be specified.{}'.format(os.linesep))

    loginfo('Setting {} processor cores for the MKL library ...'.format(args.num_cores))
    set_mkl(args.num_cores)

    config = Utils.read_config(args.configuration)
    # TODO try to load normalization
    norm = None
    try:
        plda = PLDA(config['PLDA']['model_path'])
    except IOError:
        logwarning('PLDA model initialization failed. Cosine distance will be used instead.')
        plda = None
    diar = Diarization(args.input_list, args.ivecs_dir, args.out_dir, norm, plda)
    scores = diar.score_ivec()
    if args.out_dir is not None:
        diar.dump_rttm(scores)

