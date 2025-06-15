#!/bin/bash
#OAR -q production
#OAR -p cluster='grele'
#OAR -l /nodes=1,walltime=24
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error
# Exit on error
set -e
set -o pipefail

if [ -f "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=true conda activate dictation_dev
fi

id=0,1,2,3

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/envs/dictation_dev/lib"

export PYTHONPATH=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall:$PYTHONPATH

download_path=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets
release=cv-corpus-13.0-2023-03-09
# lhotse download commonvoice --languages en --release $release $download_path
manifests=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/data/manifests
lhotse prepare commonvoice --language en -j 15 $download_path/$release $manifests
# max_dur=850 # graffiti
# max_dur=1200 # grat

# exp_dir=exp/common

# exp_dir=tmp
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u train_mux.py --world-size 2 --num-epochs 40 --start-epoch 1 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode_e2e.py --mode-id 0 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model
