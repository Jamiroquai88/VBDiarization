# vbdiar

Speaker diarization based on x-vectors using pretrained model trained in Kaldi (https://github.com/kaldi-asr/kaldi) 
and converted to ONNX format (https://github.com/onnx/onnx) running in ONNXRuntime (https://github.com/Microsoft/onnxruntime).

X-vector model was trained using VoxCeleb1 and VoxCeleb2 16k data (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about).

If you make use of the code or model, cite this: https://www.vutbr.cz/en/students/final-thesis/detail/122072
 

## Dependencies

Dependencies are listed in `requirements.txt`.

## Installation

It is recommended to use anaconda environment https://www.anaconda.com/download/.
Run `python setup.py install`
Also, since we are using Kaldi, path to Kaldi root must be set in `vbdiar/kaldi/__init__.py`

## Configs

Config file declares used models and paths to them. Example configuration file is `configs/vbdiar.yml`.

## Models

Pretrained models are stored in `models/` directory.

## Examples

Example script `examples/diarization.py` is able to run full diarization process. The code is designed in a way, that you have everything in same tree structure with relative paths in list and then you just specify directories - audio, VAD, output, etc. See example configuration.

### Required Arguments

`'-l', '--input-list'` - specifies relative path to files for testing, it is possible to specify number of speakers as the second column. Do not use file suffixes, path is always relative to input directory and suffix. 

`'-c', '--configuration'` - specifies configuration file/

`'-m', '--mode'` - specifies running mode, there are two possible modes, classic `diarization` mode which should segment 
    utterance into speakers and `sre` mode used for speaker recognition, which runs clustering for N iterations and saves all clusters

### Non-required Arguments

`'--audio-dir'` - directory with audio files in `.wav` format - `8000Hz, 16bit-s, 1c`.

`'--vad-dir'` - directory with lab files - Voice/Speech activity detection - format `speech_start speech_end`.

`'--in-emb-dir'` - input directory containing embeddings (if they were previously saved).

`'--out-emb-dir'` - output directory for storing embeddings.

`'--norm-list'` - input list with files for score normalization. When performing score normalization, it is necessary to use input ground truth `.rttm` files with unique speaker label. Speaker labels should not overlap, only in case, that there is same speaker in more audio files. All normalization utterances will be merged by speaker labels.

`'--in-rttm-dir'` - input directory with `.rttm` files (used primary for score normalization)

`'--out-rttm-dir'` - output directory for storing `.rttm` files

`'--min-window-size'` - minimal size of embedding window in miliseconds. Defines minimal size used for clustering algorithms.

`'--max-window-size'` - maximal size of embedding window in miliseconds.

`'--vad-tolerance'` - skip `n` frames of non-speech and merge them as speech.

`'--max-num-speakers'` - maximal number of speakers. Used in clustering algorithm.

`'--use-gpu'` - use GPU instead of cpu (onnxruntime-gpu must be installed)


## Results on Datasets

### AMI corpus http://groups.inf.ed.ac.uk/ami/corpus/ (development and evaluation set together)
It is important to note that these results are obtained using summed individual head-mounted microphones. 
Results are reporting when using oracle number of speakers, collar size 0.25s and without scoring overlapped speech.
Data were upsampled from 8k to 16k and 8k wav data are no longer supported.

Results can be obtained using similar command
```bash
python diarization.py -c ../configs/vbdiar.yml -l lists/AMI_dev-eval.scp --audio-dir wav/AMI/IHM_SUM --vad-dir vad/AMI --out-emb-dir emb/AMI/IHM_SUM --in-rttm-dir rttms/AMI
```

| System                                                                 | DER   |
|------------------------------------------------------------------------|-------|
| Oracle number of speakers + x-vectors + mean + LDA + L2 Norm + GPLDA   | 6.67  |
| Oracle number of speakers + x-vectors + mean + LDA + L2 Norm           | 9.16  |
| x-vectors + mean + LDA + L2 Norm + GPLDA                               | 15.54 |
