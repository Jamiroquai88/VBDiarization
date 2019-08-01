python diarization.py -c ../configs/vbdiar.yml \
    -l lists/list_spk.scp \
    --audio-dir wav/fisher-english-p1 \
     --vad-dir vad/fisher-english-p1 \
     --mode diarization \
     --out-emb-dir embeddings
