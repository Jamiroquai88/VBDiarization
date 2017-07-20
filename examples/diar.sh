#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$CDIR/../python/diar.py -l $CDIR/lists/testme_spk.scp --ivecs-dir $CDIR/i-vectors --out-dir $CDIR/rttm \
    --plda-model-dir $CDIR/../models