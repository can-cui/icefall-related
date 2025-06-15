
import logging
import string
import re
import unidecode

from pathlib import Path
from lhotse import load_manifest
from num2words import num2words

punc_1 = r"[\ʼ\ʻ\ʽ\ʾ\†\ː\ʿ\‘\„\×\®\§\«\»\…\"\!\#\$\%\&\(\)\*\+\/\<\=\>\?\@\[\]^\_\{\|\}\~\¡]" # skipping \'\’\`
punc_2 = r"[\\…\"\#\$\%\&\(\)\*\+\/\<\=\>\@\[\]^\_\{\|\}\~\→]" # skipping \'\’\`
punc_3 = r"(\'|\’|\`) "
punc_4 = r" (\'|\’|\`)"
punc_5 = r"(\'\'|\’\’|\`\`)"
punc_6= r"(\–\—)"

def preprocess_cv():
    src_dir = Path("data/fbank")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = (
        "dev",
        "test",
        "train",
    )

    logging.info("processing manifests")
    for partition in dataset_parts:
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"cv-en_cuts_{partition}_clean.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        cut_set = load_manifest(src_dir / f"cv-en_cuts_{partition}.jsonl.gz")
        for cut in cut_set:
            line =  cut.supervisions[0].text
            if "\t" in line:
            	#print(line)
            	line = line.split('\t')[0]
            	#print(line)
            #line = line.translate(str.maketrans('', '', string.punctuation))
            line = re.sub(punc_5, ' ', line)
            line = re.sub(punc_1, ' ', line)
            line = re.sub(punc_2, ' ', line)
            line = re.sub(punc_3, ' ', line)
            line = re.sub(punc_4, ' ', line)
            line = re.sub(punc_6, '-', line)
            line = n2w(line)

            line = line.replace("’", "'")
            line = line.replace("`", "'")
            if line[-1] == "'":
            	line = line[:-1]
            if line[0] == "'":
            	line = line[1:]
            
            line = unidecode.unidecode(line)
            line = line.replace(" @ ", " <unk> ")
            # line = line.upper()
            
            cut.supervisions[0].text = " ".join(line.split())
            cut.supervisions[0].custom = {"origin": "cv-en"}

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def n2w(corpus_clean):
    ordinals_th = re.findall(r"( ?[0-9]+th ?)", corpus_clean)
    for ordinal_th in ordinals_th:
        corpus_clean = re.sub(f"{ordinal_th}", ' '+num2words(ordinal_th.strip()[:-2], lang='en', to="ordinal")+' ', corpus_clean)
        
    ordinals_st = re.findall(r"( ?[0-9]+st ?)", corpus_clean)
    for ordinal_st in ordinals_st:
        corpus_clean = re.sub(f"{ordinal_st}", ' '+num2words(ordinal_st.strip()[:-2], lang='en', to="ordinal")+' ', corpus_clean)

    ordinals_nd = re.findall(r"( ?[0-9]+nd ?)", corpus_clean)
    for ordinal_nd in ordinals_nd:
        corpus_clean = re.sub(f"{ordinal_nd}", ' '+num2words(ordinal_nd.strip()[:-2], lang='en', to="ordinal")+' ', corpus_clean)

    ordinals_rd = re.findall(r"( ?[0-9]+rd ?)", corpus_clean)
    for ordinal_rd in ordinals_rd:
        corpus_clean = re.sub(f"{ordinal_rd}", ' '+num2words(ordinal_rd.strip()[:-2], lang='en', to="ordinal")+' ', corpus_clean)

    m = re.findall(r"( ?[0-9]+s ?)", corpus_clean)
    for num in m:
    	if num[-1] == "y":
        	corpus_clean = re.sub(f"{num}", ' '+num2words(num.strip()[:-1], lang='en')+'ies ', corpus_clean)
    	else:
        	corpus_clean = re.sub(f"{num}", ' '+num2words(num.strip()[:-1], lang='en')+'s ', corpus_clean)

    m = re.findall(r"( ?[0-9]+am ?)", corpus_clean)
    for num in m:
        corpus_clean = re.sub(f"{num}", ' '+num2words(num.strip()[:-2], lang='en')+' am ', corpus_clean)

    m = re.findall(r"( ?[0-9]+pm ?)", corpus_clean)
    for num in m:
        corpus_clean = re.sub(f"{num}", ' '+num2words(num.strip()[:-2], lang='en')+' pm ', corpus_clean)

    m = re.findall(r"( ?[0-9]+ ?)", corpus_clean)
    for num in m:
        corpus_clean = re.sub(f"{num}", ' '+num2words(num, lang='en')+' ', corpus_clean)
    return corpus_clean

def main():
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess_cv()


if __name__ == "__main__":
    main()
