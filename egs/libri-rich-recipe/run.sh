#!/bin/bash
#OAR -q production
#OAR -p cluster='grele'
#OAR -l /nodes=1,walltime=96
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

# waiting list:
# max_dur=1200 # gruss

max_dur=850 # graffiti
# max_dur=1200 # grat

# common rich 4279874 4280619 4282913 4284610
# exp_dir=lstm_transducer_stateless3/exp-6lstm-bert-common-rich
# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train.py --world-size 4 --num-epochs 70 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode.py --exp-dir $exp_dir --decoder rich --epoch 70 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model

# # common uncase 4279920 4280398 4282324
# exp_dir=lstm_transducer_stateless3_uncase/exp-6lstm-bert-common-rich
# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_uncase.py --world-size 4 --num-epochs 70 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_uncase.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode.py --decoder common --exp-dir $exp_dir --epoch 68 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_uncase.model

# common + uncase + rich 4279925 (wrong)/ 4282073 4283316
# exp_dir=lstm_transducer_stateless3_bertuncase_case/exp-6lstm-bert-common-rich
# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_uncase_case.py --world-size 4 --num-epochs 70 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode_uncase_case.py --decoder common --exp-dir $exp_dir --epoch 70 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model

# save to onnx
# python ./lstm_transducer_stateless3_ctc/export_onnx.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_rich.model --epoch 40 --avg 5 --num-encoder-layers 6

# # common + rich 1-decoder
# max_dur=900 # graffiti / grele
# # exp_dir=E2E_decoder_5_5/exp-6lstm-bert-common-rich # 4283969
# # exp_dir=E2E_decoder_8_2/exp-6lstm-bert-common-rich # 4284140
# exp_dir=E2E_decoder_2_8/exp-6lstm-bert-common-rich # 4284141
# exp_dir=E2E_decoder_35_65/exp-6lstm-bert-common-rich # 4284148 4285420
# # exp_dir=E2E_decoder_65_35/exp-6lstm-bert-common-rich # 4284147 4285419
# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_e2e.py --prop-common 0.35 --scaled-loss False --ctc-loss-scale 0.3 --world-size 4 --num-epochs 50 --start-epoch 28 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# exp_dir=exp/E2E_decoder_8_2_scaled/exp-6lstm-bert-common-rich # 4286437 end
# exp_dir=exp/E2E_decoder_65_35_scaled/exp-6lstm-bert-common-rich # 4286443 end
# exp_dir=exp/E2E_decoder_5_5_scaled/exp-6lstm-bert-common-rich # 4286444
# exp_dir=exp/E2E_decoder_35_65_scaled/exp-6lstm-bert-common-rich # 4286446
# exp_dir=exp/E2E_decoder_2_8_scaled/exp-6lstm-bert-common-rich # 4286451
# # exp_dir=tmp
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_e2e.py --prop-common 0.2 --ctc-loss-scale 0.3 --world-size 2 --num-epochs 40 --start-epoch 1 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode_e2e.py --mode-id 1 --exp-dir $exp_dir --epoch 33 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model
python calculate_wer/unpunc_silero.py
