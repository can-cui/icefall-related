#!/bin/bash
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l /nodes=1,walltime=96
# # File where prompts will be outputted
#OAR -O OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E OUT/oar_job.%jobid%.error
# Exit on error
set -e
set -o pipefail

# if [ -f "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh" ]; then
#     . "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/etc/profile.d/conda.sh"
#     CONDA_CHANGEPS1=true conda activate dictation_dev
# fi

# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/envs/dictation_dev/lib"

# export PYTHONPATH=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall:$PYTHONPATH

########## for export ONNX ##########
. /home/ccui/miniconda3/etc/profile.d/conda.sh
CONDA_CHANGEPS1=true conda activate virEnv_py39
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/ccui/miniconda3/envs/virEnv_py39/lib"
export PYTHONPATH=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
##########

id=0,1,2,3

# download_path=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets
# release=cv-corpus-13.0-2023-03-09
# # lhotse download commonvoice --languages en --release $release $download_path
# manifests=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/data/manifests
# lhotse prepare commonvoice --language en -j 15 $download_path/$release $manifests

## compute filterbank for cv
# python local/compute_fbank_commonvoice_en.py # 4290236

## clean labels
# python update_cv_cuts.py

## generate text
# zcat data/fbank/librispeech_cuts_train-clean-100_rich.jsonl.gz data/fbank/librispeech_cuts_train-clean-360_rich.jsonl.gz data/fbank/librispeech_cuts_train-other-500_rich.jsonl.gz data/fbank/cv-en_cuts_train_clean.jsonl.gz data/fbank/cv-en_cuts_dev_clean.jsonl.gz data/fbank/ami-ihm_cuts_train.jsonl.gz data/fbank/ami-ihm_cuts_dev.jsonl.gz | sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g | grep -P "text\:" | cut -d "\"" -f 2 | sort -u >data/lang_bpe_500/mux-rich-transcript_words.txt

## train BPE rich model
# python3 -u local/train_bpe_model.py --lang-dir data/lang_bpe_500/ --transcript data/lang_bpe_500/mux-rich-transcript_words.txt --vocab-size 500 2>&1 | tee train_bpe_mux_rich.log

## generate common labels
# python local/normalize.py
# python -u local/train_bpe_model.py --lang-dir data/lang_bpe_500/ --transcript data/lang_bpe_500/mux-common-transcript_words.txt --vocab-size 500 2>&1 | tee train_bpe_mux_com.log

# max_dur=850 # graffiti
# max_dur=1200 # grat

max_dur=900
# exp_dir=tmp
# exp_dir=exp/train_mux_rich # 4297825 4298244 4305934
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python -u train_mux/train_mux.py --world-size 4 --num-epochs 40 --start-epoch 18 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_mux_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode_e2e.py --mode-id 0 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model
exp_dir=exp/train_mux_common # 4291150 4305957
# export to onnx
# python ./lstm_transducer_stateless2_stream-mod_for_ctc_noproj/export_onnx.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_mux_rich.model --epoch 40 --avg 5 --num-encoder-layers 6
# python ./train_mux/export_onnx.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_mux_common.model --epoch 40 --avg 5 --num-encoder-layers 6
python ./lstm_transducer_stateless2_stream-mod_for_ctc_noproj/export_onnx_stream.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_rich.model --epoch 40 --avg 5 --num-encoder-layers 6

# exp_dir=exp/train_mux_common # 4291150 4305957
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python -u train_mux/train_mux_common.py --world-size 4 --num-epochs 40 --start-epoch 9 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_mux_common.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./train_mux/decode_e2e.py --mode-id 0 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./lstm_transducer_stateless2_stream-mod_for_ctc_noproj/export_onnx.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_mux_common.model --epoch 40 --avg 5 --num-encoder-layers 6
