#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
"""
This script loads ONNX models and uses them to decode waves.
You can use the following command to get the exported models:

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --onnx 1

Usage of this script:

./lstm_transducer_stateless2/onnx-streaming-decode.py \
  --encoder-model-filename ./lstm_transducer_stateless2/exp/encoder.onnx \
  --decoder-model-filename ./lstm_transducer_stateless2/exp/decoder.onnx \
  --joiner-model-filename ./lstm_transducer_stateless2/exp/joiner.onnx \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  /path/to/foo.wav \
  /path/to/bar.wav
"""

import argparse
import logging
from typing import List, Optional, Tuple

import onnxruntime as ort
import sentencepiece as spm
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

import numpy  # for debug
import time
from rpunct import RestorePuncts
from transformers import BertTokenizer, BertModel
import sys

# from onnx_pretrained import greedy_search, OnnxModel

sys.path.append(
    "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/BertPuncCap"
)
from model import BertPuncCap

import os

os.environ["TRANSFORMERS_CACHE"] = (
    "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui"
)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "sound_filename",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="Context size of the decoder model",
    )

    return parser.parse_args()


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. " f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


class Model:
    def __init__(self, args):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 5
        session_opts.intra_op_num_threads = 5
        self.session_opts = session_opts

        self.init_encoder(args)
        self.init_decoder(args)
        self.init_joiner(args)

    def init_encoder(self, args):
        self.encoder = ort.InferenceSession(
            args.encoder_model_filename,
            sess_options=self.session_opts,
        )

    def init_decoder(self, args):
        self.decoder = ort.InferenceSession(
            args.decoder_model_filename,
            sess_options=self.session_opts,
        )

    def init_joiner(self, args):
        self.joiner = ort.InferenceSession(
            args.joiner_model_filename,
            sess_options=self.session_opts,
        )

    def run_encoder(self, x, x_lens) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (N, T, C)
          h0:
            A tensor of shape (num_layers, N, proj_size)
          c0:
            A tensor of shape (num_layers, N, hidden_size)
        Returns:
          Return a tuple containing:
            - encoder_out: A tensor of shape (N, T', C')
            - next_h0: A tensor of shape (num_layers, N, proj_size)
            - next_c0: A tensor of shape (num_layers, N, hidden_size)
        """
        encoder_input_nodes = self.encoder.get_inputs()
        encoder_out_nodes = self.encoder.get_outputs()
        x_lens = torch.tensor([x.size(1)], dtype=torch.int64)

        # encoder_out, encoder_out_lens, ctc_out, next_h0, next_c0 = self.encoder.run(
        #     [
        #         encoder_out_nodes[0].name,
        #         encoder_out_nodes[1].name,
        #         encoder_out_nodes[2].name,
        #         encoder_out_nodes[3].name,
        #         encoder_out_nodes[4].name,
        #     ],
        #     {
        #         encoder_input_nodes[0].name: x.numpy(),
        #         encoder_input_nodes[1].name: x_lens.numpy(),
        #         encoder_input_nodes[2].name: h0.numpy(),
        #         encoder_input_nodes[3].name: c0.numpy(),
        #     },
        # )
        # return (
        #     torch.from_numpy(encoder_out),
        #     torch.from_numpy(next_h0),
        #     torch.from_numpy(next_c0),
        # )
        encoder_out, encoder_out_lens = self.encoder.run(
            [
                encoder_out_nodes[0].name,
                encoder_out_nodes[1].name,
            ],
            {
                encoder_input_nodes[0].name: x.numpy(),
                encoder_input_nodes[1].name: x_lens.numpy(),
            },
        )

        # encoder_out = self.encoder_proj.run(
        #     [
        #         encoder_out_nodes[2].name,
        #     ],
        #     {
        #         encoder_input_nodes[2].name: encoder_out.numpy(),
        #     },
        # )
        # Now encoder_out is of shape (N, T, joiner_dim)

        return torch.from_numpy(encoder_out), torch.from_numpy(encoder_out_lens)

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A tensor of shape (N, context_size). Its dtype is torch.int64.
        Returns:
          Return a tensor of shape (N, 1, decoder_out_dim).
        """
        decoder_input_nodes = self.decoder.get_inputs()
        decoder_output_nodes = self.decoder.get_outputs()

        decoder_out = self.decoder.run(
            [decoder_output_nodes[0].name],
            {
                decoder_input_nodes[0].name: decoder_input.numpy(),
            },
        )[0]

        return torch.from_numpy(decoder_out)
        # need_pad = False
        # decoder_output = self.decoder(y, need_pad=need_pad)
        # decoder_output = decoder_output.squeeze(1)
        # output = self.decoder_proj(decoder_output)

        # return output

    def run_joiner(
        self,
        projected_encoder_out: torch.Tensor,
        projected_decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          projected_encoder_out:
            A tensor of shape (N, joiner_dim)
          projected_decoder_out:
            A tensor of shape (N, joiner_dim)
        Returns:
          Return a tensor of shape (N, vocab_size)
        """
        joiner_input_nodes = self.joiner.get_inputs()
        joiner_output_nodes = self.joiner.get_outputs()

        logits = self.joiner.run(
            [joiner_output_nodes[0].name],
            {
                joiner_input_nodes[0].name: projected_encoder_out.numpy(),
                joiner_input_nodes[1].name: projected_decoder_out.numpy(),
                # joiner_input_nodes[2].name: torch.tensor(1.0).numpy(),
            },
        )[0]

        return torch.from_numpy(logits)
        # logit = projected_encoder_out + projected_decoder_out
        # logit = self.output_linear(torch.tanh(logit))
        # return logit


def create_streaming_feature_extractor() -> OnlineFeature:
    """Create a CPU streaming feature extractor.

    At present, we assume it returns a fbank feature extractor with
    fixed options. In the future, we will support passing in the options
    from outside.

    Returns:
      Return a CPU streaming feature extractor.
    """
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    return OnlineFbank(opts)


def greedy_search(
    model: Model,
    encoder_out: torch.Tensor,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
):
    assert encoder_out.ndim == 2
    assert encoder_out.shape[0] == 1, "TODO: support batch_size > 1"
    context_size = 2
    blank_id = 0

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)  # (1, context_size)
        decoder_out = model.run_decoder(decoder_input)
    else:
        assert decoder_out.shape[0] == 1
        assert hyp is not None, hyp

    joiner_out = model.run_joiner(encoder_out, decoder_out)
    y = joiner_out.squeeze(0).argmax(dim=0).item()

    if y != blank_id:
        hyp.append(y)
        decoder_input = hyp[-context_size:]
        decoder_input = torch.tensor([decoder_input], dtype=torch.int64)
        decoder_out = model.run_decoder(decoder_input)

    return hyp, decoder_out


def main():
    rpunct = RestorePuncts(use_cuda=False)  # BERT
    # # rpunct = RestorePuncts()
    # model_silero_te, example_texts, languages, punct, apply_te = torch.hub.load(
    #     repo_or_dir="snakers4/silero-models", model="silero_te"
    # )  # Light
    ## fr
    # BERT_name = "bert-base-multilingual-cased"
    # bert_tokenizer = BertTokenizer.from_pretrained(BERT_name)
    # bert_model = BertModel.from_pretrained(BERT_name)
    # checkpoint_path = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/BertPuncCap/models/mbert_base_cased_fr"
    # bert_punc_cap = BertPuncCap(bert_model, bert_tokenizer, checkpoint_path)

    all_time = 0
    args = get_args()
    logging.info(vars(args))

    model = Model(args)

    sound_file = args.sound_filename
    sample_rate = 16000

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # logging.info(f"Reading sound files: {sound_file}")
    # wave_samples_list = read_sound_files(
    #     filenames=sound_file,
    #     expected_sample_rate=sample_rate,
    # )

    num_encoder_layers = 6
    batch_size = 1
    d_model = 512
    rnn_hidden_size = 1024
    context_size = 2

    zero_padding = torch.zeros(8000, dtype=torch.float)

    # for index,wave_samples in enumerate(wave_samples_list):
    for index, wave in enumerate(sound_file):
        start_time = time.time()
        wave_samples = read_sound_files(
            filenames=[wave],
            expected_sample_rate=sample_rate,
        )[0]
        wave_samples = torch.cat([zero_padding, wave_samples, zero_padding], dim=0)
        # h0 = torch.zeros(num_encoder_layers, batch_size, d_model)
        # c0 = torch.zeros(num_encoder_layers, batch_size, rnn_hidden_size)

        logging.info("Constructing Fbank computer")
        online_fbank = create_streaming_feature_extractor()

        hyp = None
        decoder_out = None

        num_processed_frames = 0
        segment = 9
        offset = 4

        chunk = 3200  # 0.2 second

        start = 0
        all_feats = []
        all_enc_out = []
        while start < wave_samples.numel():
            end = min(start + chunk, wave_samples.numel())
            samples = wave_samples[start:end]
            start += chunk

            online_fbank.accept_waveform(
                sampling_rate=sample_rate,
                waveform=samples,
            )
            while online_fbank.num_frames_ready - num_processed_frames >= segment:
                frames = []
                for i in range(segment):
                    one_frame = online_fbank.get_frame(num_processed_frames + i)
                    frames.append(one_frame)
                    all_feats.append(one_frame)
                num_processed_frames += offset
                frames = torch.cat(frames, dim=0).unsqueeze(0)
                # encoder_out, h0, c0 = model.run_encoder(frames, h0, c0)
                # all_enc_out.append(encoder_out[0, :, :])
                # hyp, decoder_out = greedy_search(
                #     model, encoder_out.squeeze(0), decoder_out, hyp
                # )
                # logging.info(sp.decode(hyp[context_size:]))
                encoder_out, encoder_out_lens = model.run_encoder(
                    x=frames, x_lens=frames.size(0)
                )
                hyp, decoder_out = greedy_search(
                    model, encoder_out.squeeze(0), decoder_out, hyp
                )
                logging.info(sp.decode(hyp[context_size:]))

        online_fbank.input_finished()
        while online_fbank.num_frames_ready - num_processed_frames >= segment:
            frames = []
            for i in range(segment):
                one_frame = online_fbank.get_frame(num_processed_frames + i)
                frames.append(one_frame)
            num_processed_frames += offset
            frames = torch.cat(frames, dim=0).unsqueeze(0)
            # encoder_out, h0, c0 = model.run_encoder(frames, h0, c0)
            # all_enc_out.append(encoder_out[0, :, :])
            # hyp, decoder_out = greedy_search(
            #     model, encoder_out.squeeze(0), decoder_out, hyp
            # )
            encoder_out, encoder_out_lens = model.run_encoder(
                x=frames, x_lens=frames.size(0)
            )
            hyp, decoder_out = greedy_search(
                model, encoder_out.squeeze(0), decoder_out, hyp
            )
        row_text = sp.decode(hyp[context_size:])
        # try:
        out = rpunct.punctuate(row_text.lower(), lang="en")  # BERT
        #     out = apply_te(row_text.lower(), lan="en")  # light
        #     # out=bert_punc_cap.predict(row_text)
        # except:
        # out = row_text
        end_time = time.time()
        processing_time = end_time - start_time

        all_time += processing_time
        logging.info(sound_file[index])
        logging.info(out)
        logging.info(processing_time)
    logging.info("All processing time is:%f", all_time)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
