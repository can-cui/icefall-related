#!/bin/bash
#OAR -q production
#OAR -p cluster='grue'
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

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train.py --world-size 4 --num-epochs 40 --start-epoch 1 --exp-dir lstm_transducer_stateless3_ctc/exp-ami-scratch/ --max-duration 750 --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe.model --use-fp16 1 --ctc-loss-scale 0.3 2>&1 | tee lstm_transducer_stateless3_ctc/exp-ami-scratch/train.log
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_ori.py --world-size 4 --num-epochs 40 --start-epoch 1 --exp-dir lstm_transducer_stateless3_ctc/exp-ami-scratch/ --max-duration 750 --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe.model --use-fp16 1 2>&1 | tee lstm_transducer_stateless3_ctc/exp-ami-scratch/train.log
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_noCtc.py --world-size 2 --num-epochs 40 --start-epoch 30 --exp-dir lstm_transducer_stateless3/exp-ami-scratch-noCtc-cond/ --max-duration 750 --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe.model --use-fp16 1 --ctc-loss-scale 0 2>&1 | tee lstm_transducer_stateless3/exp-ami-scratch-noCtc-cond/train.log
