# Copyright    2022  Xiaomi Corp.        (authors: Zengwei Yao)
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

import copy
import math
from typing import List, Optional, Tuple

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledLinear,
    ScaledLSTM,
)
from torch import nn

LOG_EPSILON = math.log(1e-10)


class RNNOnline(EncoderInterface):
    """
    Args:
      num_features (int):
        Number of input features.
      subsampling_factor (int):
        Subsampling factor of encoder (convolution layers before lstm layers) (default=4).  # noqa
      d_model (int):
        Output dimension (default=512).
      dim_feedforward (int):
        Feedforward dimension (default=2048).
      rnn_hidden_size (int):
        Hidden dimension for lstm layers (default=1024).
      num_encoder_layers (int):
        Number of encoder layers (default=12).
      dropout (float):
        Dropout rate (default=0.1).
      layer_dropout (float):
        Dropout value for model-level warmup (default=0.075).
      aux_layer_period (int):
        Period of auxiliary layers used for random combiner during training.
        If set to 0, will not use the random combiner (Default).
        You can set a positive integer to use the random combiner, e.g., 3.
    """

    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 3,
        d_model: int = 500,
        dim_feedforward: int = 500,
        rnn_hidden_size: int = 500,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        aux_layer_period: int = 0,
    ) -> None:
        super(RNNOnline, self).__init__()

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 3:
            raise NotImplementedError("Support only 'subsampling_factor=3'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T+2)//3, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T+2)//3
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv1dSubsampling(num_features, d_model)

        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.rnn_hidden_size = rnn_hidden_size

        encoder_layer = RNNEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            rnn_hidden_size=rnn_hidden_size,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )
        assert aux_layer_period == 0, "aux_layer_period !=0"
        self.encoder = RNNEncoder(
            encoder_layer,
            num_encoder_layers,
        )


    def forward(self, x: torch.Tensor, f_cache: torch.Tensor, tdnn_cache: torch.Tensor, lstm_cntxts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, C, T), where N is the batch size,
            T is the sequence length, C is the feature dimension.
        """
        x, tdnn_out = self.encoder_embed(x, f_cache, tdnn_cache)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        assert not self.training

        h = lstm_cntxts[:, 0, 0, :, :] # (ll, N, C)
        c = lstm_cntxts[:, 1, 0, :, :] # (ll, N, C)
        x, new_h, new_c = self.encoder(x, h, c)
        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        
        new_h = new_h[:,None,None,:,:] # (ll,1,1,N,H)
        new_c = new_c[:,None,None,:,:] # (ll,1,1,N,H)
        lstm_cntxts[:, 0] = new_h[:, 0]
        lstm_cntxts[:, 1] = new_c[:, 0]		
		
        return x, tdnn_out, lstm_cntxts


class RNNEncoderLayer(nn.Module):
    """
    RNNEncoderLayer is made up of lstm and feedforward networks.

    Args:
      d_model:
        The number of expected features in the input (required).
      dim_feedforward:
        The dimension of feedforward network model (default=2048).
      rnn_hidden_size:
        The hidden dimension of rnn layer.
      dropout:
        The dropout value (default=0.1).
      layer_dropout:
        The dropout value for model-level warmup (default=0.075).
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        rnn_hidden_size: int,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
    ) -> None:
        super(RNNEncoderLayer, self).__init__()
        self.layer_dropout = layer_dropout
        self.d_model = d_model
        self.rnn_hidden_size = rnn_hidden_size

        assert rnn_hidden_size >= d_model, (rnn_hidden_size, d_model)
        self.lstm = ScaledLSTM(
            input_size=d_model,
            hidden_size=rnn_hidden_size,
            proj_size=d_model if rnn_hidden_size > d_model else 0,
            num_layers=1,
            dropout=0.0,
        )
        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )
        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).  # noqa
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          h:
            The hidden states of all layers,
            with shape of (1, N, d_model);
          c:
            The cell states of all layers,
            with shape of (1, N, rnn_hidden_size).
        """
        src_orig = src

        alpha = 1.0

        # lstm module
        assert not self.training
        if not torch.jit.is_tracing():
            # for hidden state
            assert h.shape == (1, src.size(1), self.d_model)
            # for cell state
            assert c.shape == (1, src.size(1), self.rnn_hidden_size)
        src_lstm, new_h, new_c = self.lstm(src, h, c)
        src = self.dropout(src_lstm) + src

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src, new_h, new_c


class RNNEncoder(nn.Module):
    """
    RNNEncoder is a stack of N encoder layers.

    Args:
      encoder_layer:
        An instance of the RNNEncoderLayer() class (required).
      num_layers:
        The number of sub-encoder-layers in the encoder (required).
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
    ) -> None:
        super(RNNEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = encoder_layer.d_model
        self.rnn_hidden_size = encoder_layer.rnn_hidden_size

    def forward(
        self,
        src: torch.Tensor,
        h: torch.Tensor, 
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass the input through the encoder layer in turn.

        Args:
          src:
            The sequence to the encoder layer (required).
            Its shape is (S, N, E), where S is the sequence length,
            N is the batch size, and E is the feature number.
          h:
            The hidden states of all layers,
            with shape of (num_layers, N, d_model);
          c:
          The cell states of all layers,
            with shape of (num_layers, N, rnn_hidden_size).
        """
        assert not self.training
        if not torch.jit.is_tracing():
            # for hidden state
            assert h.shape == (
                self.num_layers,
                src.size(1),
                self.d_model,
            ), h.shape
            # for cell state
            assert c.shape == (
                self.num_layers,
                src.size(1),
                self.rnn_hidden_size,
            )

        output = src
        new_hidden_states = []
        new_cell_states = []

        for i, mod in enumerate(self.layers):
            cur_h = h[i : i + 1, :, :]  # h: (1, N, d_model)
            cur_c = c[i : i + 1, :, :]  # c: (1, N, rnn_hidden_size)
            output, new_h, new_c = mod(output, cur_h, cur_c)
            new_hidden_states.append(new_h)
            new_cell_states.append(new_c)

        return output, torch.cat(new_hidden_states, dim=0), torch.cat(new_cell_states, dim=0)


class Conv1dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 500,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, in_channels, T).
          out_channels
            Output dim. The output shape is (N, (T+2)//3, out_channels)
        """
        assert in_channels >= 9
        super().__init__()

        self.tdnn_1 = nn.Sequential(
            ScaledConv1d( #1
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_2 = nn.Sequential(
            ScaledConv1d( #2
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_3 = nn.Sequential(
            ScaledConv1d( #3
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_4 = nn.Sequential(
            ScaledConv1d( #4
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_5 = nn.Sequential(
            ScaledConv1d( #5
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_6 = nn.Sequential(
            ScaledConv1d( #6 #len=T remains unchanged until here
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.tdnn_7 = nn.Sequential(
            ScaledConv1d( #7
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=3, # sub_sampling factor
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(
            out_channels, out_channels
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(out_channels, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

    def forward(self, f: torch.Tensor, f_cache: torch.Tensor, tdnn_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
          x:
            Its shape is (N, idim, T).

        Returns:
          Return a tensor of shape (N, (T+2)//3, odim)
        """
        tdnn_out = []
        f = torch.cat((f_cache, f), 2)
        tdnn_1_y = self.tdnn_1(f)
        tdnn_1_y = torch.cat((tdnn_cache[0], tdnn_1_y), 2)
        tdnn_out.append(tdnn_1_y)
        tdnn_2_y = self.tdnn_2(tdnn_1_y)
        tdnn_2_y = torch.cat((tdnn_cache[1], tdnn_2_y), 2)
        tdnn_out.append(tdnn_2_y)
        tdnn_3_y = self.tdnn_3(tdnn_2_y)
        tdnn_3_y = torch.cat((tdnn_cache[2], tdnn_3_y), 2)
        tdnn_out.append(tdnn_3_y)
        tdnn_4_y = self.tdnn_4(tdnn_3_y)
        tdnn_4_y = torch.cat((tdnn_cache[3], tdnn_4_y), 2)
        tdnn_out.append(tdnn_4_y)
        tdnn_5_y = self.tdnn_5(tdnn_4_y)
        tdnn_5_y = torch.cat((tdnn_cache[4], tdnn_5_y), 2)
        tdnn_out.append(tdnn_5_y)
        tdnn_6_y = self.tdnn_6(tdnn_5_y)
        tdnn_7_y = self.tdnn_7(tdnn_6_y)
        tdnn_out = torch.stack(tdnn_out, dim=0)[:,:,:,-2:]
        
        x = tdnn_7_y.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.out(x)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        
        return x, tdnn_out



if __name__ == "__main__":
    feature_dim = 80
    m = RNN(
        num_features=feature_dim,
        d_model=512,
        rnn_hidden_size=1024,
        dim_feedforward=2048,
        num_encoder_layers=12,
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = m(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )
    num_param = sum([p.numel() for p in m.parameters()])
    print(f"Number of model parameters: {num_param}")
