#! /usr/bin/env python

import sys
import argparse

from wav2ivecs import set_mkl
from lib.diarization import Diarization


def main(argv):
    set_mkl()

    parser = argparse.ArgumentParser('Run diarization on input data with PLDA model.')
    parser.add_argument('-l', '--input-list', help='list of input files without suffix',
                        action='store', dest='input_list', type=str, required=True)
    parser.add_argument('--ivecs-dir', help='directory containing i-vectos using IvecSet pickle format',
                        action='store', dest='ivecs_dir', type=str, required=True)
    parser.add_argument('--out-dir', help='output directory for storing .rttm files',
                        action='store', dest='out_dir', type=str, required=True)
    parser.add_argument('--plda-model-dir', help='directory with PLDA model files',
                        action='store', dest='plda_model_dir', type=str, required=True)

    args = parser.parse_args()

    diar = Diarization(args.input_list, args.ivecs_dir, args.out_dir,  args.plda_model_dir)
    scores = diar.score()
    # diar.get_der(scores, args.segments_file, True)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
