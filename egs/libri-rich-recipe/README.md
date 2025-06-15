- Prepare librispeech data
```
mkdir -p downloads
mkdir -p data/manifests
lhotse download librispeech --full downloads/
lhotse prepare librispeech -j 10 downloads/LibriSpeech data/manifests
lhotse prepare librispeech -j 10 /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech data/manifests
mkdir -p data/fbank
python local/compute_fbank_librispeech.py
```

- Download and prepare data for data augmentation (only if it was not done before)
```
lhotse download musan downloads/ # noise for data augmentation
lhotse prepare musan downloads/musan data/manifests
python local/compute_fbank_musan.py
```

- Update to rich trnascripts and train BPE model (already done!)
```
zcat data/fbank/librispeech_cuts_train-clean-100_rich.jsonl.gz data/fbank/librispeech_cuts_train-clean-360_rich.jsonl.gz data/fbank/librispeech_cuts_train-other-500_rich.jsonl.gz |  sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g |  grep -P "text\:" | cut -d "\"" -f 2 | sort -u > data/lang_bpe_500/bert-rich-filtered-transcript_words.txt 

python3 -u local/train_bpe_model.py --lang-dir data/lang_bpe_500/ --transcript data/lang_bpe_500/book-rich-uncase-filtered-transcript_words.txt --vocab-size 500 2>&1 | tee train_bpe_book-rich-uncase-filtered.log

python3 -u update_librispeech_cuts.py
```

- Launch training
CUDA_VISIBLE_DEVICES="0,1" python3 -u lstm_transducer_stateless3_ctc/train.py --world-size 2 --num-epochs 2 --start-epoch 1  --exp-dir lstm_transducer_stateless3/exp-6lstm-bert-common-rich/ --max-duration 750 --num-encoder-layers 6 --bpe-model data/lang_bpe_500/bpe.model --use-fp16 1  2>&1 | tee lstm_transducer_stateless3/exp-6lstm-bert-common-rich/train.log


rsync --exclude .git --exclude __pycache__ --exclude .DS_Store -avzhP . nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe

scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/lstm_transducer_stateless3/exp-6lstm-bert-common-rich/tensorboard/ /Users/ccui/Desktop/dictation_result/libri_common_rich/

zcat data/fbank/librispeech_cuts_train-clean-100_rich_uncase.jsonl.gz data/fbank/librispeech_cuts_train-clean-360_rich_uncase.jsonl.gz data/fbank/librispeech_cuts_train-other-500_rich_uncase.jsonl.gz |  sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g |  grep -P "text\:" | cut -d "\"" -f 2 | sort -u > data/lang_bpe_500/bert-rich-filtered-uncase_words.txt 

python3 -u local/train_bpe_model.py --lang-dir data/lang_bpe_500/ --transcript data/lang_bpe_500/bert-rich-filtered-uncase_words.txt --vocab-size 500 2>&1 | tee train_bpe_bert-rich-uncase-filtered.log

scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/lstm_transducer_stateless3/exp-6lstm-bert-common-rich/tensorboard/ /Users/ccui/Desktop/dictation_result/libri_common_rich

scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/lstm_transducer_stateless3_uncase/exp-6lstm-bert-common-rich/tensorboard/ /Users/ccui/Desktop/dictation_result/libri_common_uncase

scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/paper-libri-rich/ASR/exp/E2E_dec_5_5/tensorboard/ /Users/ccui/Desktop/dictation_result/zip-book

zcat data/fbank/librispeech_cuts_train-clean-100_book_uncase.jsonl.gz data/fbank/librispeech_cuts_train-clean-360_book_uncase.jsonl.gz data/fbank/librispeech_cuts_train-other-500_book_uncase.jsonl.gz |  sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g |  grep -P "text\:" | cut -d "\"" -f 2 | sort -u > data/lang_bpe_500/book-rich-uncase-filtered-transcript_words.txt 

scp nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/paper-libri-rich/ASR/data/uncase_data.py /Users/ccui/Desktop/icefall/paper-libri-rich/ASR/data

scp nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/lstm_transducer_stateless3_uncase/exp-6lstm-bert-common-rich/greedy_search/recogs /Users/ccui/Desktop/icefall/libri-rich-recipe/calculate_wer/greedy_search_common_uncase 

zcat data/fbank/librispeech_cuts_train-clean-100.jsonl.gz data/fbank/librispeech_cuts_train-clean-360.jsonl.gz data/fbank/librispeech_cuts_train-other-500.jsonl.gz |  sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g |  grep -P "text\:" | cut -d "\"" -f 2 | sort -u > data/lang_bpe_500/common-transcript_words.txt 

python3 -u local/train_bpe_model.py --lang-dir data/lang_bpe_500/ --transcript data/lang_bpe_500/common-transcript_words.txt --vocab-size 500 2>&1 | tee train_bpe_common.log

zcat data/fbank/librispeech_cuts_train-clean-100_rich.jsonl.gz data/fbank/librispeech_cuts_train-clean-360_rich.jsonl.gz data/fbank/librispeech_cuts_train-other-500_rich.jsonl.gz data/fbank/cv-en_cuts_train.jsonl.gz data/fbank/cv-en_cuts_dev.jsonl.gz data/fbank/ami-ihm_cuts_train.jsonl.gz data/fbank/ami-ihm_cuts_dev.jsonl.gz |  sed -e s/\"text\"\:/\\ntext:/g | sed -e s/\"language\"\:/\\nlanguage:/g |  grep -P "text\:" | cut -d "\"" -f 2 | sort -u > data/lang_bpe_500/bert-rich-filtered-transcript_words.txt 

scp nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/exp/train_mux_rich/joiner-epoch-40-avg-5.uint8-quant.onnx /Users/ccui/Desktop/dictation_result/train_mux_rich
