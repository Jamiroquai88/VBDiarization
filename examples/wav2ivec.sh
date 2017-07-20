#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$CDIR/../python/wav2ivec.py -l $CDIR/lists/testme.scp --audio-dir $CDIR/wav --vad-dir $CDIR/vad \
    --out-dir $CDIR/i-vectors --ubm-file $CDIR/../models/GMM.npy --v-file $CDIR/../models/v600_iter10.npy
