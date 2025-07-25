#!/usr/bin/env python3
# flake8: noqa
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script converts several saved checkpoints
# to a single one using model averaging.
"""

Usage:

(1) Export to torchscript model using torch.jit.trace()

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 35 \
  --avg 10 \
  --jit-trace 1

It will generate 3 files: `encoder_jit_trace.pt`,
`decoder_jit_trace.pt`, and `joiner_jit_trace.pt`.

(2) Export `model.state_dict()`

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 35 \
  --avg 10

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

To use the generated file with `lstm_transducer_stateless2/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./lstm_transducer_stateless2/decode.py \
        --exp-dir ./lstm_transducer_stateless2/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
    # You will find the pre-trained models in icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp

(3) Export to ONNX format

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --onnx 1

It will generate the following files in the given `exp_dir`.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx

Please see ./streaming-onnx-decode.py for usage of the generated files

Check
https://github.com/k2-fsa/sherpa-onnx
for how to use the exported models outside of icefall.
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace.
        It will generate 3 files:
         - encoder_jit_trace.pt
         - decoder_jit_trace.pt
         - joiner_jit_trace.pt

        Check ./jit_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--pnnx",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace for later
        converting to PNNX. It will generate 3 files:
         - encoder_jit_trace-pnnx.pt
         - decoder_jit_trace-pnnx.pt
         - joiner_jit_trace-pnnx.pt
        """,
    )

    parser.add_argument(
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit and --pnnx are ignored and it exports the model
        to onnx format. It will generate the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx

        Refer to ./onnx_check.py and ./onnx_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def export_encoder_model_jit_trace(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.trace()

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)
    states = encoder_model.get_init_states()

    traced_model = torch.jit.trace(encoder_model, (x, x_lens, states))
    traced_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_trace(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.trace()

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = torch.tensor([False])

    traced_model = torch.jit.trace(decoder_model, (y, need_pad))
    traced_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_trace(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.

    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    traced_model = torch.jit.trace(joiner_model, (encoder_out, decoder_out))
    traced_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


def export_encoder_model_onnx(
    encoder_model: nn.Module,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has 3 inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64
        - states: a tuple containing:
            - h0: a tensor of shape (num_layers, N, proj_size)
            - c0: a tensor of shape (num_layers, N, hidden_size)

    and it has 3 outputs:

        - encoder_out, a tensor of shape (N, T, C)
        - encoder_out_lens, a tensor of shape (N,)
        - states: a tuple containing:
            - next_h0: a tensor of shape (num_layers, N, proj_size)
            - next_c0: a tensor of shape (num_layers, N, hidden_size)

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    N = 1
    T = 100 #requires: T >= 9
    x = torch.zeros(N, T, 80, dtype=torch.float32)
    x_lens = torch.tensor([T], dtype=torch.int64)
    h = torch.rand(encoder_model.num_encoder_layers, N, encoder_model.d_model)
    c = torch.rand(
        encoder_model.num_encoder_layers, N, encoder_model.rnn_hidden_size
    )

    scripted_encoder_model = torch.jit.script(encoder_model)

    torch.onnx.export(
        scripted_encoder_model,
        #encoder_model,  # use torch.jit.trace() internally
        (x, x_lens, h, c),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens", "h", "c"],
        output_names=["encoder_out", "encoder_out_lens", "next_h", "next_c"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "h": {1: "N"},
            "c": {1: "N"},
            "encoder_out": {0: "N", 1: "T"},
            "encoder_out_lens": {0: "N"},
            "next_h": {1: "N"},
            "next_c": {1: "N"},
        },
    )
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_onnx(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, 1, C)

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = False  # Always False, so we can use torch.jit.trace() here
    # Note(fangjun): torch.jit.trace() is more efficient than torch.jit.script()
    # in this case
    torch.onnx.export(
        decoder_model,
        (y, need_pad),
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y", "need_pad"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_filename}")


def export_ctc_model_onnx(
    ctc_model: nn.Module,
    ctc_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the ctc model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, encoder_dim)

    and has one output:

        - ctc_out: a torch.float32 tensor of shape (N, vocab_size)

    Args:
      ctc_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    encoder_out = torch.zeros(10, ctc_model[1].weight.shape[1], dtype=torch.float32)
    torch.onnx.export(
        ctc_model,
        (encoder_out),
        ctc_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["ctc_out"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "ctc_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {ctc_filename}")


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:

        - projected_encoder_out: a tensor of shape (N, joiner_dim)
        - projected_decoder_out: a tensor of shape (N, joiner_dim)

    and produces one output:

        - logit: a tensor of shape (N, vocab_size)

    The exported encoder_proj model has one input:

        - encoder_out: a tensor of shape (N, encoder_out_dim)

    and produces one output:

        - projected_encoder_out: a tensor of shape (N, joiner_dim)

    The exported decoder_proj model has one input:

        - decoder_out: a tensor of shape (N, decoder_out_dim)

    and produces one output:

        - projected_decoder_out: a tensor of shape (N, joiner_dim)
    """
    encoder_proj_filename = str(joiner_filename).replace(
        ".onnx", "_encoder_proj.onnx"
    )

    decoder_proj_filename = str(joiner_filename).replace(
        ".onnx", "_decoder_proj.onnx"
    )

    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    joiner_dim = joiner_model.decoder_proj.weight.shape[0]

    projected_encoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)

    project_input = False
    # Note: It uses torch.jit.trace() internally
    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out, project_input),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "projected_encoder_out",
            "projected_decoder_out",
            "project_input",
        ],
        output_names=["logit"],
        dynamic_axes={
            "projected_encoder_out": {0: "N"},
            "projected_decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")

    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.encoder_proj,
        encoder_out,
        encoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["projected_encoder_out"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "projected_encoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {encoder_proj_filename}")

    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.decoder_proj,
        decoder_out,
        decoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["decoder_out"],
        output_names=["projected_decoder_out"],
        dynamic_axes={
            "decoder_out": {0: "N"},
            "projected_decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_proj_filename}")


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    #if torch.cuda.is_available():
    #    device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    if params.pnnx:
        params.is_pnnx = params.pnnx
        logging.info("For PNNX")

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device),
                strict=False,
            )
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device),
                strict=False,
            )
    else:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg + 1]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )

    model.to("cpu")
    model.eval()

    if params.onnx:
        logging.info("Export model to ONNX format")
        convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

        opset_version = 11
        encoder_filename = params.exp_dir / "encoder.onnx"
        export_encoder_model_onnx(
            model.encoder,
            encoder_filename,
            opset_version=opset_version,
        )

        decoder_filename = params.exp_dir / "decoder.onnx"
        export_decoder_model_onnx(
            model.decoder,
            decoder_filename,
            opset_version=opset_version,
        )

        joiner_filename = params.exp_dir / "joiner.onnx"
        export_joiner_model_onnx(
            model.joiner,
            joiner_filename,
            opset_version=opset_version,
        )

        ctc_filename = params.exp_dir / "ctc_decoder.onnx"
        export_ctc_model_onnx(
            model.ctc_model,
            ctc_filename=ctc_filename,
            opset_version=opset_version,
        )

    elif params.pnnx:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace-pnnx.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace-pnnx.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace-pnnx.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    elif params.jit_trace is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    else:
        logging.info("Not using torchscript")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
