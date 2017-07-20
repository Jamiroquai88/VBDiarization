#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np


def main(argv):
    parser = argparse.ArgumentParser('Convert txt.gz representation of matrices to npy.')
    parser.add_argument('-d', '--input-dir', help='input directory with models',
                        action='store', dest='input_dir', type=str, required=True)
    args = parser.parse_args()

    d = args.input_dir
    files = ['GMM', 'v600_iter10']
    for f in files:
        np.save(os.path.join(d, f), np.loadtxt(os.path.join(d, '{}.txt.gz'.format(f))))
    files = ['backend/backend.LDA', 'backend/backend.mu_train', 'backend/backend.PLDA.Gamma',
             'backend/backend.PLDA.Lambda', 'backend/backend.PLDA.c', 'backend/backend/PLDA.c']
    for f in files:
        np.save(os.path.join(d, f.replace('backend/', '')), np.loadtxt(os.path.join(d, '{}.txt.gz'.format(f))))

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
