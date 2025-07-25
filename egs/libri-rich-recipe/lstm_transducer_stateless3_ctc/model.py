# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


from typing import Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder_common: nn.Module,
        decoder_rich: nn.Module,
        joiner_common: nn.Module,
        joiner_rich: nn.Module,
        # ctc_model: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
          ctc_model:
            Model for the CTC part.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder_common, "blank_id")
        assert hasattr(decoder_rich, "blank_id")

        self.encoder = encoder
        self.decoder_common = decoder_common
        self.decoder_rich = decoder_rich
        self.joiner_common = joiner_common
        self.joiner_rich = joiner_rich


        self.simple_am_proj = ScaledLinear(
            encoder_dim, vocab_size, initial_speed=0.5
        )
        self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size)
        # self.ctc_model = ctc_model

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y_common: k2.RaggedTensor,
        y_rich: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        warmup: float = 1.0,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
        Returns:
          Return a tuple containing:
            - The loss for the "trivial" joiner
            - The loss for the non-linear joiner
            - CTC loss

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y_common.num_axes == 2, y_common.num_axes
        assert y_rich.num_axes == 2, y_rich.num_axes

        assert x.size(0) == x_lens.size(0) == y_common.dim0

        encoder_out, x_lens, _ = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        def get_loss(encoder_out, x_lens, y,decoder,joiner):
          # Now for the decoder, i.e., the prediction network
          row_splits = y.shape.row_splits(1)
          y_lens = row_splits[1:] - row_splits[:-1]

          blank_id = decoder.blank_id
          sos_y = add_sos(y, sos_id=blank_id)

          # sos_y_padded: [B, S + 1], start with SOS.
          sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

          # decoder_out: [B, S + 1, decoder_dim]
          decoder_out = decoder(sos_y_padded)

          # Note: y does not start with SOS
          # y_padded : [B, S]
          y_padded = y.pad(mode="constant", padding_value=0)

          y_padded = y_padded.to(torch.int64)
          boundary = torch.zeros(
              (x.size(0), 4), dtype=torch.int64, device=x.device
          )
          boundary[:, 2] = y_lens
          boundary[:, 3] = x_lens

          lm = self.simple_lm_proj(decoder_out)
          am = self.simple_am_proj(encoder_out)

          with torch.cuda.amp.autocast(enabled=False):
              simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                  lm=lm.float(),
                  am=am.float(),
                  symbols=y_padded,
                  termination_symbol=blank_id,
                  lm_only_scale=lm_scale,
                  am_only_scale=am_scale,
                  boundary=boundary,
                  reduction=reduction,
                  return_grad=True,
              )

          # ranges : [B, T, prune_range]
          ranges = k2.get_rnnt_prune_ranges(
              px_grad=px_grad,
              py_grad=py_grad,
              boundary=boundary,
              s_range=prune_range,
          )

          # am_pruned : [B, T, prune_range, encoder_dim]
          # lm_pruned : [B, T, prune_range, decoder_dim]
          am_pruned, lm_pruned = k2.do_rnnt_pruning(
              am=joiner.encoder_proj(encoder_out),
              lm=joiner.decoder_proj(decoder_out),
              ranges=ranges,
          )

          # logits : [B, T, prune_range, vocab_size]

          # project_input=False since we applied the decoder's input projections
          # prior to do_rnnt_pruning (this is an optimization for speed).
          logits = joiner(am_pruned, lm_pruned, project_input=False)

          with torch.cuda.amp.autocast(enabled=False):
              pruned_loss = k2.rnnt_loss_pruned(
                  logits=logits.float(),
                  symbols=y_padded,
                  ranges=ranges,
                  termination_symbol=blank_id,
                  boundary=boundary,
                  reduction=reduction,
              )

          # calculate ctc loss
          # nnet_output = self.ctc_model(encoder_out)

          targets = []
          target_lengths = []
          for t in y.tolist():
              target_lengths.append(len(t))
              targets.extend(t)

          targets = torch.tensor(
              targets,
              device=x.device,
              dtype=torch.int64,
          )

          target_lengths = torch.tensor(
              target_lengths,
              device=x.device,
              dtype=torch.int64,
          )

          # ctc_loss = torch.nn.functional.ctc_loss(
          #     log_probs=nnet_output.permute(1, 0, 2),  # (T, N, C)
          #     targets=targets,
          #     input_lengths=x_lens,
          #     target_lengths=target_lengths,
          #     reduction="none",
          # )
          
          return (simple_loss, pruned_loss)
        
        simple_loss_common, pruned_loss_common=get_loss(encoder_out, x_lens, y_common,self.decoder_common,self.joiner_common)
        simple_loss_rich, pruned_loss_rich=get_loss(encoder_out, x_lens, y_rich,self.decoder_rich,self.joiner_rich)
        return (simple_loss_common, pruned_loss_common,simple_loss_rich, pruned_loss_rich)
