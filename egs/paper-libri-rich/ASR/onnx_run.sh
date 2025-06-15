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

# MODEL_DIREC="/Users/ccui/Desktop/dictation/libri-bok-rich-trans-stream-decode"
# MODEL_DIREC="/Users/ccui/Desktop/dictation/bert_model"
# MODEL_DIREC="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/bert_model" # home_made_audio 29.374588 TTS 4266353 test-clean 4266354 test_other_wav 4266375 AMI 4266376
# MODEL_DIREC="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/libri-only" #
# MODEL_DIREC="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/libri-bok-rich-trans-stream-decode" # 4264197
# MODEL_DIREC="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/6lstm_joint"
## model mux

# Directory containing the WAV files
# WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/home_made_audio" # book 4264249 only_bert only_light freespeech 4269316 mux_rich 4308839 mux_common 4309158
# WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/TTS/data" # book 4266401 only_bert 4266338 only_light 4266343 freespeech 4269317 mux_rich 4308843 mux_common 4309159
# WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/test_clean/wav" # common 4636456 book 4638350 bert 4637271 common_book (common) 4638392 common_bert 4638395 E2E_book (common) 4638382 E2E_book (rich) 4638385 E2E_bert(common) 4638387 E2E_bert(rich) 4638391
WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/test_other_wav" # common 4636399 book 4637845 bert 4637367 common_book (common) 4638393 common_bert 4638394 E2E_book (common) 4638383 E2E_book (rich) 4638384 E2E_bert(common) 4638388 E2E_bert(rich) 4638389
# WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/AMI_test/full-corpus-asr_test-segments" # book 4266499 only_bert 4266332 only_light 4266351 freespeech 4269320 mux_rich 4309155 mux_common 4309175
# WAV_DIRECTORY="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/CommonVoice/test" # book 4267163 bert 4267164 only_bert 4267165 only_light 4267166 freespeech 4269321 mux_rich 4309156 mux_common 4309179

# Output directory for TXT files
# OUTPUT_FILE="libri_onlfvc'(ry.txt"

# Iterate over each WAV file in the directory
# for file in "$WAV_DIRECTORY"/*.wav; do
#     # Get the filename without extension
#     filename=$(basename "$file" .wav)

#     # Process the file using cheetah_demo_file
#     # output=$(python3 -u $MODEL_DIREC/streaming-joint-onnx-decode.py --encoder-model-filename $MODEL_DIREC/uint8-quant-onnx/encoder.onnx --decoder-model-filename $MODEL_DIREC/uint8-quant-onnx/decoder.onnx --joiner-model-filename $MODEL_DIREC/uint8-quant-onnx/joiner.onnx --bpe-model $MODEL_DIREC/bpe.model $file)
#     python3 -u $MODEL_DIREC/streaming-joint-onnx-decode.py --encoder-model-filename $MODEL_DIREC/uint8-quant-onnx/encoder.onnx --decoder-model-filename $MODEL_DIREC/uint8-quant-onnx/decoder.onnx --joiner-model-filename $MODEL_DIREC/uint8-quant-onnx/joiner.onnx --bpe-model $MODEL_DIREC/bpe.model $file
#     # echo "${filename},${output}" >>"$OUTPUT_FILE"
#     # echo "${output}" >>"$OUTPUT_FILE"

#     # echo "Processed $file. Appended output to $OUTPUT_FILE"
# done
# python3 -u $MODEL_DIREC/streaming-joint-onnx-decode.py --encoder-model-filename $MODEL_DIREC/uint8-quant-onnx/encoder.onnx --decoder-model-filename $MODEL_DIREC/uint8-quant-onnx/decoder.onnx --joiner-model-filename $MODEL_DIREC/uint8-quant-onnx/joiner.onnx --bpe-model $MODEL_DIREC/bpe.model /Users/ccui/Desktop/dictation/TTS/data/alice_d22-2.wav
# Use the find command to get a list of WAV files in the directory
WAV_FILES=$(find "$WAV_DIRECTORY" -name "*.wav" -type f)
# echo "${WAV_FILES}"
# # Join the WAV files using a backslash \
# WAV_FILES_JOINED=$(echo "$WAV_FILES" | tr '\n' '\\')

# python3 /Users/ccui/Desktop/dictation/streaming-joint-onnx-decode.py \
# ONNX=$MODEL_DIREC/uint8-quant-onnx
# # ONNX=$MODEL_DIREC/onnx
# python3 /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/dictation/streaming-joint-onnx-decode.py \
#     --encoder-model-filename $ONNX/encoder.onnx \
#     --decoder-model-filename $ONNX/decoder.onnx \
#     --joiner-model-filename $ONNX/joiner.onnx \
#     --bpe-model $MODEL_DIREC/bpe.model \
#     $WAV_FILES
MODEL_DIREC="exp/common"
epo=40 #(56 is 1: rich)
## mux models
# BPE="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/data/lang_bpe_500/bpe_mux_rich.model"
BPE="data/lang_bpe_500/bpe_common.model"
python3 streaming-joint-onnx-decode.py \
    --encoder-model-filename $MODEL_DIREC/encoder-epoch-$epo-avg-5.onnx \
    --decoder-model-filename $MODEL_DIREC/decoder-epoch-$epo-avg-5.onnx \
    --joiner-model-filename $MODEL_DIREC/joiner-epoch-$epo-avg-5.onnx \
    --bpe-model $BPE \
    $WAV_FILES
