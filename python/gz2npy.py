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
    gmm = np.loadtxt(os.path.join(d, 'GMM.txt.gz'))
    np.save(os.path.join(d, 'GMM'), gmm)
    v = np.loadtxt(os.path.join(d, 'v600_iter10.txt.gz'))
    np.save(os.path.join(d, 'v600_iter10'), v)
    lda_file = np.loadtxt(os.path.join(d, 'backend/backend.LDA.txt.gz'))
    np.save(os.path.join(d, 'backend.LDA'), lda_file)
    mu_file = np.loadtxt(os.path.join(d, 'backend/backend.mu_train.txt.gz'))
    np.save(os.path.join(d, 'backend.mu_train'), mu_file)
    gamma_file = np.loadtxt(os.path.join(d, 'backend/backend.PLDA.Gamma.txt.gz'))
    np.save(os.path.join(d, 'backend.PLDA.Gamma'), gamma_file)
    lambda_file = np.loadtxt(os.path.join(d, 'backend/backend.PLDA.Lambda.txt.gz'))
    np.save(os.path.join(d, 'backend.PLDA.Lambda'), lambda_file)
    c_file = np.loadtxt(os.path.join(d, 'backend/backend.PLDA.c.txt.gz'))
    np.save(os.path.join(d, 'backend.PLDA.c'), c_file)
    k_file = np.loadtxt(os.path.join(d, 'backend/backend.PLDA.k.txt.gz'))
    np.save(os.path.join(d, 'backend.PLDA.k'), k_file)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
