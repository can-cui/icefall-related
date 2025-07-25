from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMP(nn.Module):
    """LSTM with projection.

    PyTorch does not support exporting LSTM with projection to ONNX.
    This class reimplements LSTM with projection using basic matrix-matrix
    and matrix-vector operations. It is not intended for training.
    """

    def __init__(self, lstm: nn.LSTM):
        """
        Args:
          lstm:
            LSTM with proj_size. We support only uni-directional,
            1-layer LSTM with projection at present.
        """
        super().__init__()
        assert lstm.bidirectional is False, lstm.bidirectional
        assert lstm.num_layers == 1, lstm.num_layers
        
        assert 0 < lstm.proj_size < lstm.hidden_size, ( # comment to handle proj_size==0
            lstm.proj_size,
            lstm.hidden_size,
        )

        assert lstm.batch_first is False, lstm.batch_first

        state_dict = lstm.state_dict()

        w_ih = state_dict["weight_ih_l0"]
        w_hh = state_dict["weight_hh_l0"]

        b_ih = state_dict["bias_ih_l0"]
        b_hh = state_dict["bias_hh_l0"]

        '''
        if lstm.proj_size !=0:
          w_hr = state_dict["weight_hr_l0"]
          self.proj_size = lstm.proj_size
        else:
          w_hr = torch.eye(lstm.hidden_size)
          self.proj_size = lstm.hidden_size
        '''
        
        w_hr = state_dict["weight_hr_l0"]
        self.proj_size = lstm.proj_size
        
        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size

        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b = b_ih + b_hh
        self.w_hr = w_hr

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor, 
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          input:
            A tensor of shape [T, N, hidden_size]
          h0: 
            a tensor of shape (1, N, proj_size)
          c0: 
            a tensor of shape (1, N, hidden_size)
        Returns:
          Return a tuple containing:
            - output: a tensor of shape (T, N, proj_size).
            - h: a tensor of shape (1, N, proj_size)
            - c: a tensor of shape (1, N, hidden_size)

        """
        h0 = h.squeeze(0)
        c0 = c.squeeze(0)
        y_list = []

        for t in range(input.shape[0]):
            x = input[t,:,:]
            gates = F.linear(x, self.w_ih, self.b) + F.linear(h0, self.w_hh)
            i, f, g, o = gates.chunk(4, dim=1)

            i = i.sigmoid()
            f = f.sigmoid()
            g = g.tanh()
            o = o.sigmoid()

            c = f * c0 + i * g
            h = o * c.tanh()

            h = F.linear(h, self.w_hr)
            y_list.append(h)

            c0 = c
            h0 = h

        y = torch.stack(y_list, dim=0)

        return y, h0.unsqueeze(0), c0.unsqueeze(0)
