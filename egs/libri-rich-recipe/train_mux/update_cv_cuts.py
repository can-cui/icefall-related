
import logging
import string, re
from pathlib import Path

from lhotse import load_manifest
from num2words import num2words
from cvutils import Validator

punc_1 = r"[\ʼ\ʻ\ʽ\ʾ\†\ː\ʿ\‘\“\”\„\×\®\§\«\»\…\"\!\#\$\%\&\(\)\*\+\,\.\/\:\;\<\=\>\?\@\[\]^\_\{\|\}\~]" # skipping \'\’\`\-\–\—
punc_2 = r"[\\…\"\!\#\$\%\&\(\)\*\+\,\.\/\:\;\<\=\>\?\@\[\]^\_\{\|\}\~]" # skipping \'\’\`\-\–\—
punc_3 = r"(\'|\’|\`|\-|\–|\—) "
punc_4 = r" (\'|\’|\`|\-|\–|\—)"

def preprocess_cv():
    src_dir = Path("data/fbank")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)
    
    v = Validator('fr')

    dataset_parts = (
        "dev",
        "test",
        "train",
    )

    logging.info("processing manifests")
    for partition in dataset_parts:
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"cv-fr_cuts_{partition}.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        cut_set = load_manifest(src_dir / f"cv-fr_cuts_{partition}_orig.jsonl.gz")
        for cut in cut_set:
            line =  cut.supervisions[0].text
            if "\t" in line:
            	line = line.split('\t')[0]

            # line = line.translate(str.maketrans('', '', string.punctuation)) # skip this for FR
            line = re.sub(punc_1, ' ', line)
            line = re.sub(punc_2, ' ', line)
            line = re.sub(punc_3, ' ', line)
            line = re.sub(punc_4, ' ', line)
            line = n2w(line)
            line= line.strip()
            
            piece_list = []
            for piece in line.split():
            	vpiece = v.validate(piece)
            	if not vpiece:
            	    vpiece = piece
            	piece_list.append(vpiece)
            
            line = " ".join(piece_list) 
            line = line.strip().upper()

            line = line.replace("Ü", "U")
            line = line.replace("Ÿ", "Y")
            line = line.replace("Æ", "AE")
            
            cut.supervisions[0].text = line
            cut.supervisions[0].custom = {"origin": "cv-fr"}

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def n2w(line):
    line = line.replace("1er", "premier")
    line = line.replace("1ere", "première")
    line = line.replace("1ère", "première")
    line = line.replace("2nde", "seconde")
    line = line.replace("2nd", "seconde")
            
    ordinals_th = re.findall(r"( ?[0-9]+ème ?)", line)
    for ordinal_th in ordinals_th:
        line = re.sub(f"{ordinal_th}", ' '+num2words(ordinal_th.strip()[:-3], lang='fr', to="ordinal")+' ', line)

    m = re.findall(r"( ?[0-9]+ ?)", line)
    for num in m:
        line = re.sub(f"{num}", ' '+num2words(num, lang='fr')+' ', line)
        line = re.sub(f"{num}", ' '+num2words(num, lang='fr')+' ', line)
    return line
    

def main():
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess_cv()


if __name__ == "__main__":
    main()
