#!/usr/bin/env python3
# Copyright    2022  Vivoka        
#
# Based on https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compute_fbank_librispeech.py


"""
This file computes fbank features of the mls_french dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import lhotse

lhotse.set_audio_duration_mismatch_tolerance(0.1)  # 10ms tolerance

def compute_fbank():
    src_dir = Path("data/manifests")#manifests")
    output_dir = Path("data/fbank")#fbank")
    num_jobs = min(7, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        "dev",
        "test",
        "train",
    )

    prefix = "cv-en"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins, sampling_rate=16000))
    # extractor = KaldifeatFbank(KaldifeatFbankConfig())

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            if (output_dir / f"{prefix}_cuts_{partition}.{suffix}").is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            cut_set = cut_set.resample(16000)
            if "train" in partition:
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cur_num_jobs = num_jobs if ex is None else 7
            #cur_num_jobs = min(cur_num_jobs, len(cut_set))

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=cur_num_jobs,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            # Split long cuts into many short and un-overlapping cuts
            cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
            cut_set.to_file(output_dir / f"{prefix}_cuts_{partition}.{suffix}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank()
