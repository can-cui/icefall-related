import logging

from pathlib import Path
from functools import lru_cache
from lhotse import CutSet, load_manifest_lazy


class CV:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cv-nl_cuts_dev.jsonl.gz
                - cv-nl_cuts_test.jsonl.gz
                - cv-nl_cuts_train.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv-nl_cuts_train.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv-nl_cuts_dev.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv-nl_cuts_test.jsonl.gz"
        )
