#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang)

"""
This script exports a transducer model from PyTorch to ONNX.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-iter-1224000-avg-14.pt"

cd exp
ln -s pretrained-iter-1224000-avg-14.pt epoch-9999.pt
popd

2. Export the model to ONNX

./pruned_transducer_stateless3/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 9999 \
  --avg 1 \
  --exp-dir $repo/exp/

It will generate the following 3 files inside $repo/exp:

  - encoder-epoch-9999-avg-1.onnx
  - decoder-epoch-9999-avg-1.onnx
  - joiner-epoch-9999-avg-1.onnx

See ./onnx_pretrained.py and ./onnx_check.py for how to
use the exported ONNX models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import sentencepiece as spm
import torch
import torch.nn as nn
from lstm import RNN
from decoder import Decoder
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import setup_logger
from onnxruntime.quantization import quantize_dynamic, QuantType

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
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxEncoder(nn.Module):
    """A wrapper for LSTM and the encoder_proj from the joiner"""

    def __init__(self, encoder: RNN, encoder_proj: nn.Linear, ctc_model: nn.Module):
        """
        Args:
          encoder:
            LSTM encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = encoder_proj
        self.ctc_model = ctc_model

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Please see the help information of RNN.forward

        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 1-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a tuple containing:
            - encoder_out, A 3-D tensor of shape (N, T', joiner_dim)
            - encoder_out_lens, A 1-D tensor of shape (N,)
        """
        encoder_out, encoder_out_lens, new_h, new_c = self.encoder(x, x_lens, h, c)

        ctc_output = self.ctc_model(encoder_out)

        encoder_proj_out = self.encoder_proj(encoder_out)
        # Now encoder_out is of shape (N, T, joiner_dim)

        return encoder_proj_out, encoder_out_lens, ctc_output, new_h, new_c


class OnnxDecoder(nn.Module):
    """A wrapper for Decoder and the decoder_proj from the joiner"""

    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, context_size).
        Returns
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        need_pad = False
        decoder_output = self.decoder(y, need_pad=need_pad)
        decoder_output = decoder_output.squeeze(1)
        output = self.decoder_proj(decoder_output)

        return output


class OnnxJoiner(nn.Module):
    """A wrapper for the joiner"""

    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        # temperature: float,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        logit = encoder_out + decoder_out
        logit = self.output_linear(torch.tanh(logit))
        # log_probs = (logit / temperature).log_softmax(dim=-1)
        return logit


def export_encoder_model_onnx(
    encoder_model: OnnxEncoder,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T', joiner_dim)
        - encoder_out_lens, a tensor of shape (N,)

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    N = 1
    T = 100
    D = 80
    x = torch.zeros(N, T, D, dtype=torch.float32)
    x_lens = torch.tensor([T], dtype=torch.int64)
    h = torch.rand(encoder_model.encoder.num_encoder_layers, N, encoder_model.encoder.d_model)
    c = torch.rand(encoder_model.encoder.num_encoder_layers, N, encoder_model.encoder.rnn_hidden_size)

    scripted_encoder_model = torch.jit.script(encoder_model)

    torch.onnx.export(
        #encoder_model, # has fixed context and loop in case of LSTM
        scripted_encoder_model,
        (x, x_lens, h, c),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens", "h", "c"],
        output_names=["encoder_out", "encoder_out_lens", "ctc_out", "next_h", "next_c"],
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


def export_decoder_model_onnx(
    decoder_model: OnnxDecoder,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, joiner_dim)

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size

    y = torch.zeros(10, context_size, dtype=torch.int64)
    torch.onnx.export(
        decoder_model,
        y,
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )

    meta_data = {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:

        - encoder_out: a tensor of shape (N, joiner_dim)
        - decoder_out: a tensor of shape (N, joiner_dim)

    and produces one output:

        - logit: a tensor of shape (N, vocab_size)
    """
    joiner_dim = joiner_model.output_linear.weight.shape[1]
    logging.info(f"joiner dim: {joiner_dim}")

    projected_encoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)
    # temperature = torch.ones(1, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        # (projected_encoder_out, projected_decoder_out, temperature[0]),
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out",
            "decoder_out",
            # "temperature",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    meta_data = {
        "joiner_dim": str(joiner_dim),
    }
    add_meta_data(filename=joiner_filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    setup_logger(f"{params.exp_dir}/log-export/log-export-onnx")

    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    model.to(device)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to("cpu")
    model.eval()

    print("model.parameters()", sum([p.numel() for p in model.parameters()]))
    print("model.encoder.parameters()", sum([p.numel() for p in model.encoder.parameters()]))
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
    print("scaled model.parameters()", sum([p.numel() for p in model.parameters()]))
    print("scaled model.encoder.parameters()", sum([p.numel() for p in model.encoder.parameters()]))

    encoder = OnnxEncoder(
        encoder=model.encoder,
        encoder_proj=model.joiner.encoder_proj,
        ctc_model=model.ctc_model
    )

    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )

    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    encoder.to("cpu")
    encoder.eval()
    decoder.to("cpu")
    decoder.eval()
    joiner.to("cpu")
    joiner.eval()

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"

    opset_version = 11#13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / f"encoder-{suffix}.onnx"
    export_encoder_model_onnx(
        encoder,
        encoder_filename,
        opset_version=opset_version,
    )
    logging.info(f"Exported encoder to {encoder_filename}")
    quantize_dynamic(encoder_filename, str(encoder_filename).replace('.onnx', '.uint8-quant.onnx'),  weight_type=QuantType.QUInt8, extra_options = {'EnableSubgraph': True}, optimize_model=True)

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / f"decoder-{suffix}.onnx"
    export_decoder_model_onnx(
        decoder,
        decoder_filename,
        opset_version=opset_version,
    )
    quantize_dynamic(decoder_filename, str(decoder_filename).replace('.onnx', '.uint8-quant.onnx'),  weight_type=QuantType.QUInt8, extra_options = {'EnableSubgraph': True}, optimize_model=True)

    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / f"joiner-{suffix}.onnx"
    export_joiner_model_onnx(
        joiner,
        joiner_filename,
        opset_version=opset_version,
    )
    quantize_dynamic(joiner_filename, str(joiner_filename).replace('.onnx', '.uint8-quant.onnx'),  weight_type=QuantType.QUInt8, extra_options = {'EnableSubgraph': True}, optimize_model=True)

    logging.info(f"Exported joiner to {joiner_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    main()
