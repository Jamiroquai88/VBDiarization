# vbdiar

Speaker diarization based on `Kaldi` x-vectors using pretrained model from http://kaldi-asr.org/models/0003_sre16_v2_1a.tar.gz 

## Dependencies

Dependencies are listed in `requirements.txt`.

## Installation

It is recommended to use anaconda environment https://www.anaconda.com/download/ because of mkl based implementation.
Run `python setup.py install`

## Configs

Config file declares used models and paths to them. Example configuration file is `configs/vbdiar.yml`.

## Models

Pretrained models are stored in `models/` directory.

## Examples

Example script `examples/diarization.py` is able to run full diarization process. The code is designed in a way, that you have everything in same tree structure with relative paths in list and then you just specify directories - audio, VAD, output, etc. See example configuration.

### Required Arguments

`'-l', '--input-list'` - specifies relative path to files for testing, it is possible to specify number of speakers as the second column. Do not use file suffixes, path is always relative to input directory and suffix. 

`'-c', '--configuration'` - specifies configuration file

### Non-required Arguments

`'--audio-dir'` - directory with audio files in `.wav` format - `8000Hz, 16bit-s, 1c`.

`'--vad-dir'` - directory with lab files - Voice/Speech activity detection - format `speech_start speech_end`.

`'--in-emb-dir'` - input directory containing embeddings (if they were previously saved).

`'--out-emb-dir'` - output directory for storing embeddings.

`'--norm-list'` - input list with files for score normalization. When performing score normalization, it is necessary to use input ground truth `.rttm` files with unique speaker label. Speaker labels should not overlap, only in case, that there is same speaker in more audio files. All normalization utterances will be merged by speaker labels.

`'--in-rttm-dir'` - input directory with `.rttm` files (used primary for score normalization)

`'--out-rttm-dir'` - output directory for storing `.rttm` files

`'--min-window-size'` - minimal size of i-vector window in miliseconds. Defines minimal size used for clustering algorithms.

`'--max-window-size'` - maximal size of i-vector window in miliseconds.

`'--vad-tolerance'` - skip `n` frames of non-speech and merge them as speech.

`'--max-num-speakers'` - maximal number of speakers. Used in clustering algorithm.

## Results on Datasets

### AMI corpus http://groups.inf.ed.ac.uk/ami/corpus/ (development and evaluation set together)
It is important to note that these results are obtained using summed individual head-mounted microphones. Results are reporting when using oracle number of speakers, collar size 0.25s and without scoring overlapped speech.

| System                                                           | DER   |
|------------------------------------------------------------------|-------|
| x-vectors + mean + L2 Norm                                       | 15.82 |
| x-vectors + mean + LDA + L2 Norm                                 | 15.03 |
| x-vectors + Normalization (mean and S-Norm) + L2 Norm            | 18.21 |
