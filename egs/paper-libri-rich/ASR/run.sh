#!/bin/bash
#OAR -q production
#OAR -p cluster='grappe'
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
    CONDA_CHANGEPS1=true conda activate dict
fi

# LDFLAGS="-L/Users/pzelasko/miniconda3/envs/lhotse/lib" pip install kaldifeat

id=0,1,2,3,4,5,6,7

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/miniconda3/envs/dictation_dev/lib"

export PYTHONPATH=/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall:$PYTHONPATH

# waiting list: gruss:  graffiti:  grue: grat:4310129 4311908 4311909

# max_dur=700 # gruss
# max_dur=400 # grue
# max_dur=310 # graffiti / grele
# max_dur=1200 # grat

# # book rich
max_dur=620 # graffiti
# max_dur=450 # grue
# max_dur=750 # gruss

# common
# exp_dir=exp/common # 4288294 4288715 4289956 4290612 end
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_common/train.py --world-size 4 --num-epochs 40 --start-epoch 37 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_common.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_common/decode.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_common.model
# python ./pruned_transducer_stateless7_common/decode_ami.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_common.model
# python ./pruned_transducer_stateless7_common/decode_cv.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_common.model
# python ./pruned_transducer_stateless7_common/export-onnx.py --exp-dir $exp_dir --bpe-model data/lang_bpe_500/bpe_common.model --epoch 40 --avg 5

# book rich
# exp_dir=exp/book_rich # 4281006 4282918 4284881 4286373 4288297 4288724 4289243 end
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book/train.py --world-size 4 --num-epochs 40 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_book/decode.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_book/decode_ami.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_book/decode_cv.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_book/export-onnx.py --exp-dir $exp_dir --epoch 40 --avg 5 --bpe-model data/lang_bpe_500/bpe_book.model

# # bert rich
# exp_dir=exp/bert_rich # 4281015 4282919 4285998 end
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_bert/train.py --world-size 2 --num-epochs 40 --start-epoch 9 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_bert/decode.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert/decode_ami.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert/decode_cv.py --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert/export-onnx.py --exp-dir $exp_dir --epoch 40 --avg 5 --bpe-model data/lang_bpe_500/bpe_rich.model

# common book # 4281634 4286005 end
# exp_dir=exp/common_book
# # # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_common_book/train.py --world-size 4 --num-epochs 40 --start-epoch 37 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# # python ./pruned_transducer_stateless7_common_book/decode.py --decoder common --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_common_book/decode_ami.py --decoder common --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_common_book/decode_cv.py --decoder common --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_common_book/export-onnx.py --exp-dir $exp_dir --epoch 40 --avg 5 --bpe-model data/lang_bpe_500/bpe_book.model

# common bert # 4281648 4283250 4286000 4288104 end
# exp_dir=exp/common_bert
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_common_bert/train.py --world-size 2 --num-epochs 40 --start-epoch 36 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_common_bert/decode.py --decoder rich --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_common_bert/decode_ami.py --decoder rich --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_common_bert/decode_cv.py --decoder rich --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_common_bert/export-onnx.py --exp-dir $exp_dir --epoch 40 --avg 5 --bpe-model data/lang_bpe_500/bpe_rich.model

################################ cond-dec with all the rich / common ################################
# max_dur=150 # graffiti
# max_dur=200 # grue
# max_dur=600                  # grat
# exp_dir=exp/E2E_dec_book_all # 4315569 4320388 4322739
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book_e2e_all/train.py --prop-common 0.5 --scaled-loss False --world-size 4 --num-epochs 40 --start-epoch 33 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_book_e2e/decode.py --mode-id 1 --exp-dir $exp_dir --epoch 37 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model

#bert cond-dec all
# max_dur=600
# exp_dir=exp/E2E_dec_bert_all # 4317324 4321363
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_bert_e2e_all/train.py --prop-common 0.5 --scaled-loss False --world-size 4 --num-epochs 40 --start-epoch 19 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
######################################################################################################

# bert cond-dec
# exp_dir=exp/E2E_dec_5_5_bert # 4288437 4289569 4289624 4299870
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_bert_e2e/train.py --prop-common 0.5 --scaled-loss False --world-size 4 --num-epochs 80 --start-epoch 39 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_bert_e2e/decode.py --mode-id 1 --exp-dir $exp_dir --epoch 47 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert_e2e/decode_ami.py --mode-id 1 --exp-dir $exp_dir --epoch 47 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert_e2e/decode_cv.py --mode-id 1 --exp-dir $exp_dir --epoch 47 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_rich.model
# python ./pruned_transducer_stateless7_bert_e2e/export-onnx.py --exp-dir $exp_dir --epoch 47 --avg 5 --bpe-model data/lang_bpe_500/bpe_rich.model

## non scale
# exp_dir=tmp
# max_dur=310 # grele
# max_dur=290 # 65 5 graffiti
# max_dur=450 # 65 5 grue
# exp_dir=exp/E2E_dec_95_5 # 4288319 4289197 4288763 4295312 4308623 end
# exp_dir=exp/E2E_dec_8_2 # 4286552 4287747 4288291 4288713 4289802 4290247 end
# exp_dir=exp/E2E_dec_65_35 # 4287758 4288727 4290146 4295314 end
# exp_dir=exp/E2E_dec_5_5 # 4286610 4288288 end 80epo 4289686 4290145 4290832 4299871
# exp_dir=exp/E2E_dec_35_65 # 4286658 4288306 4288725 4308619 4310795 end
# exp_dir=exp/E2E_dec_2_8 # 4288318 4308622 4310798 4311800
# exp_dir=exp/E2E_dec_5_95 #  4308807 end
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book_e2e/train.py --prop-common 0.5 --scaled-loss False --world-size 4 --num-epochs 80 --start-epoch 57 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book_e2e/train.py --prop-common 0.2 --scaled-loss False --world-size 4 --num-epochs 40 --start-epoch 38 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_book_e2e/decode.py --mode-id 1 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_book_e2e/decode_cv.py --mode-id 1 --exp-dir $exp_dir --epoch 56 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model
# python ./pruned_transducer_stateless7_book_e2e/export-onnx.py --exp-dir $exp_dir --epoch 56 --avg 5 --bpe-model data/lang_bpe_500/bpe_book.model

## scaled
# exp_dir=exp/E2E_dec_95_5_scaled # 4288289 4289626 4289803 epo50 4299909
# exp_dir=exp/E2E_dec_8_2_scaled # 4286362 4287745 4288328 4289207 4289900 epo50 4299907
# exp_dir=exp/E2E_dec_65_35_scaled # 4286986 4288714 4289953 4290609 4299880 4300858
# exp_dir=exp/E2E_dec_5_5_scaled # 4287753 4288721 to do
# exp_dir=exp/E2E_dec_35_65_scaled # 4287031 4287759
# exp_dir=exp/E2E_dec_2_8_scaled # 4287722 4288672

# exp_dir=exp/E2E_dec_95_5_scaled_v2 # 4306014 4308128 end
# exp_dir=exp/E2E_dec_8_2_scaled_v2 # 4310832 4311722 end
# exp_dir=exp/E2E_dec_65_35_scaled_v2 # 4310889 4311730 4316930 end
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book_e2e/train.py --prop-common 0.65 --world-size 4 --num-epochs 40 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_book_e2e/decode.py --mode-id 1 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model

# exp_dir=exp/E2E_dec_5_5_scaled_9 # 4288332 4288873 4289627 4289805 4290378 4295335 4300871
# exp_dir=exp/E2E_dec_5_5_scaled_8 # 4288336 4288762 4289958 4290611 4295336 end
# exp_dir=exp/E2E_dec_5_5_scaled_7 # 4288337 4288765 4289190  4291570 4295337
# exp_dir=exp/E2E_dec_5_5_scaled_6 # 4288339 4288758 4289957 4290610 4299876
# exp_dir=exp/E2E_dec_5_5_scaled_4 # 4290266 4295334 4300864
# exp_dir=exp/E2E_dec_5_5_scaled_2 # 4290268 4292386
# # # # # exp_dir=tmp
# exp_dir=exp/E2E_dec_5_5_scaled_9_v2 # 4308261 end
# exp_dir=exp/E2E_dec_5_5_scaled_8_v2 # 4309161 end
# exp_dir=exp/E2E_dec_5_5_scaled_7_v2 # 4309600 end
# exp_dir=exp/E2E_dec_5_5_scaled_6_v2 # 4310120 end
# max_dur=290 # 2 8 graffiti
# exp_dir=exp/E2E_dec_5_5_scaled_4_v2 # 4311916 4314386 4316625 4318574
# exp_dir=exp/E2E_dec_5_5_scaled_2_v2 # 4312332 4315510 4317718
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_book_e2e_scaled/train.py --scaled-value-rich 0.4 --world-size 4 --num-epochs 40 --start-epoch 34 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./pruned_transducer_stateless7_book_e2e_scaled/decode.py --mode-id 1 --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --bpe-model data/lang_bpe_500/bpe_book.model

# # common bertUncase # 4281652 4282322
# exp_dir=exp/common_bertUncase
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_common_bertUncase/train.py --world-size 2 --num-epochs 40 --start-epoch 8 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_uncase.model --use-fp16 1 2>&1 | tee $exp_dir/train.log

# # common bookUncase # 4281664 4282321 4283300
# exp_dir=exp/common_bookUncase
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u pruned_transducer_stateless7_common_bookUncase/train.py --world-size 2 --num-epochs 40 --start-epoch 17 --exp-dir $exp_dir --max-duration $max_dur --bpe-model data/lang_bpe_500/bpe_book_uncase.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode.py --exp-dir $exp_dir --epoch 50 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe.model

# # common uncase
# exp_dir=lstm_transducer_stateless3_uncase/exp-6lstm-bert-common-rich
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_uncase.py --world-size 4 --num-epochs 40 --start-epoch 4 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_uncase.model --use-fp16 1 2>&1 | tee $exp_dir/train.log

# # common + uncase + rich
# exp_dir=lstm_transducer_stateless3_uncase_case/exp-6lstm-bert-common-rich
# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$id python3 -u lstm_transducer_stateless3_ctc/train_uncase_case.py --world-size 4 --num-epochs 60 --start-epoch 40 --exp-dir $exp_dir --max-duration $max_dur --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model --use-fp16 1 2>&1 | tee $exp_dir/train.log
# python ./lstm_transducer_stateless3_ctc/decode_uncase_case.py --decoder common --exp-dir $exp_dir --epoch 40 --avg 5 --decoding-method greedy_search --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe_rich.model

python calculate_wer/uncase_silero.py
