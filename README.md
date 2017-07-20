# VBDiarization
Speaker diarization based on python implementation from http://voicebiometry.org/

## Python Dependencies

re
numpy
scipy
pickle
sklearn
subprocess
multiprocessing

## Installation
Run script in root directory - get_models.sh to download and prepare models

## Examples

### gz2npy
convert models from .txt.gz to .npy

### wav2ivec
extract i-vectors from wav audio files

### wav2ivecs
extract multiple i-vectors used for diarization from wav audio files

### diar
run diarization on previously extracted i-vector using PLDA model

