
import logging
import re
from pathlib import Path

from lhotse import load_manifest
from lhotse.cut import Cut

def preprocess():
    src_dir = Path("data/fbank")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = (
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    )

    logging.info("processing manifests")
    for partition in dataset_parts:
        logging.info(f"Processing {partition}")
        lookup = {}
        rich_transcript = f"data/bert-rich-uncase/{partition}-rich.txt"
        with open(rich_transcript) as fp:
            for line in fp:
                fields = line.strip().split(' ', 1)
                if len(fields) > 1:
                    cut_id = fields[0]
                    transcript = fields[1]
                    lookup[cut_id] = transcript
        
        new_cuts_path = output_dir / f"librispeech_cuts_{partition}_rich_uncase.jsonl.gz"

        cut_set = load_manifest(src_dir / f"librispeech_cuts_{partition}.jsonl.gz")
        for cut in cut_set:
            cut_id = cut.supervisions[0].id
            cut_id = cut_id.split('_')[0]
            cut.supervisions[0].text = lookup.get(cut_id, "")
        cut_set = cut_set.filter(remove_no_transcript)

        logging.info(f"Saving to {new_cuts_path}")
        cut_set.to_file(new_cuts_path)


def remove_no_transcript(c: Cut):
        return len(c.supervisions[0].text)>0

def main():
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess()

if __name__ == "__main__":
    main()
