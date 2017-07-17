#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

FILE="vbs_demo.tgz"
URL="http://voicebiometry.org/download/$FILE"

wget $URL $CDIR
tar -xzvf $CDIR/$FILE models
rm -f $CDIR/$FILE
$CDIR/python/gz2npy.py -d $CDIR/models
rm -rf $CDIR/models/backend
rm -f $CDIR/models/*.txt.gz
(>&1 echo "Models succesfully downloaded and installed.")

exit 0
