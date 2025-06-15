vocab_size=500

mkdir -p data/lm
cat <(gunzip -c data/manifests/ami-ihm_supervisions_train.jsonl.gz | jq '.text' | sed 's:"::g')> data/lm/transcript_words.txt

lang_dir=data/lang_bpe_${vocab_size}
mkdir -p $lang_dir

# Add special words to words.txt
echo "<eps> 0" > $lang_dir/words.txt
echo "!SIL 1" >> $lang_dir/words.txt
echo "<UNK> 2" >> $lang_dir/words.txt

# Add regular words to words.txt
cat data/lm/transcript_words.txt | grep -o -E '\w+' | sort -u | awk '{print $0,NR+2}' >> $lang_dir/words.txt

# Add remaining special word symbols expected by LM scripts.
num_words=$(cat $lang_dir/words.txt | wc -l)
echo "<s> ${num_words}" >> $lang_dir/words.txt
num_words=$(cat $lang_dir/words.txt | wc -l)
echo "</s> ${num_words}" >> $lang_dir/words.txt
num_words=$(cat $lang_dir/words.txt | wc -l)
echo "#0 ${num_words}" >> $lang_dir/words.txt

./local/train_bpe_model.py \
--lang-dir $lang_dir \
--vocab-size $vocab_size \
--transcript data/lm/transcript_words.txt
