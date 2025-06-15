import logging

from pathlib import Path
from functools import lru_cache
from lhotse import CutSet, load_manifest_lazy


class CV:
    def __init__(
        self,
    ):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cuts_dev.json.gz
                - cuts_test.json.gz
                - cuts_train.json.gz
        """
        # self.manifest_dir = Path(manifest_dir)
        self.manifest_dir = Path(
            "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/data/fbank"
        )

    # @lru_cache()
    # def train_cuts(self) -> CutSet:
    #     logging.info("About to get train cuts")
    #     return load_manifest_lazy(self.manifest_dir / "cv-en_cuts_train_clean.jsonl.gz")

    # @lru_cache()
    # def dev_cuts(self) -> CutSet:
    #     logging.info("About to get dev cuts")
    #     return load_manifest_lazy(self.manifest_dir / "cv-en_cuts_dev_clean.jsonl.gz")

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        # return load_manifest_lazy(self.manifest_dir / "cv-en_cuts_test_clean.jsonl.gz")
        return load_manifest_lazy(
            "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/libri-rich-recipe/data/fbank/cv-en_cuts_test.jsonl.gz"
        )
