import logging

from pathlib import Path
from functools import lru_cache
from lhotse import CutSet, load_manifest_lazy


class AMI:
    def __init__(self,):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cuts_dev.json.gz
                - cuts_test.json.gz
                - cuts_train.json.gz
        """
        self.manifest_dir =Path("data/fbank"),

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get AMI train cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ami-ihm_cuts_train.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get AMI dev cuts")
        return load_manifest_lazy(
            self.manifest_dir / "ami-ihm_cuts_dev.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get AMI test cuts")
        return load_manifest_lazy(
            # self.manifest_dir / "ami-ihm_cuts_test.jsonl.gz"
            "data/fbank/ami-ihm_cuts_test.jsonl.gz"
        )
