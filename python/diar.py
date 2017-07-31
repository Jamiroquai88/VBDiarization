#! /usr/bin/env python

import os
import sys
import argparse
import multiprocessing

from wav2ivecs import set_mkl
from lib.tools import loginfo
from lib.diarization import Diarization


def main(argv):
    parser = argparse.ArgumentParser('Run diarization on input data with PLDA model.')
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('--norm-list', help='list of input normalization files without suffix in .npy format',
                        action='store', dest='norm_list', type=str, required=False)
    parser.add_argument('--ivecs-dir', help='directory containing i-vectos using class IvecSet - pickle format',
                        action='store', dest='ivecs_dir', type=str, required=True)
    parser.add_argument('--out-dir', help='output directory for storing .rttm files',
                        action='store', dest='out_dir', type=str, required=False)
    parser.add_argument('--reference', help='reference rttm file for system scoring',
                        action='store', dest='reference', type=str, required=False)
    parser.add_argument('--plda-model-dir', help='directory with PLDA model files',
                        action='store', dest='plda_model_dir', type=str, required=True)
    parser.add_argument('-j', '--num-cores', help='number of processor cores to use',
                        action='store', dest='num_cores', type=int, required=False)
    parser.set_defaults(norm_list=None)
    parser.set_defaults(num_cores=multiprocessing.cpu_count())
    args = parser.parse_args()
    if args.reference is None and args.out_dir is None:
        parser.print_help()
        sys.stderr.write('at least one of --reference and --out-dir must be specified' + os.linesep)

    loginfo('[diar.main] Setting {} processor cores for the MKL library ...'.format(args.num_cores))
    set_mkl(args.num_cores)
    diar = Diarization(args.input_list, args.norm_list, args.ivecs_dir, args.out_dir,  args.plda_model_dir)
    scores = diar.score()
    if args.reference is not None:
        diar.get_der(args.reference, scores)
    if args.out_dir is not None:
        diar.dump_rttm(scores)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
