from collections import OrderedDict
from typing import Optional

import torch


class MyLSTM:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, state_dict: OrderedDict):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.wi = []
        self.wh = []
        self.bias = []

        for layer_id in range(num_layers):
            self.wi.append(state_dict[f"weight_ih_l{layer_id}"].T)
            self.wh.append(state_dict[f"weight_hh_l{layer_id}"].T)
            self.bias.append(state_dict[f"bias_ih_l{layer_id}"] + state_dict[f"bias_hh_l{layer_id}"])

    def forward(self, x, h_0: Optional[torch.Tensor] = None, c_0: Optional[torch.Tensor] = None):
        seq_len, batch_size, input_size = x.shape

        assert input_size == self.input_size

        if h_0 is None:
            h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_n = h_0.clone()

        if c_0 is None:
            c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            c_n = c_0.clone()

        out = torch.empty(seq_len, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            # layer 0
            ifgo = torch.matmul(x[t], self.wi[0]) + torch.matmul(h_n[0], self.wh[0]) + self.bias[0]

            i = ifgo[:, 0 : self.hidden_size]
            f = ifgo[:, self.hidden_size : self.hidden_size * 2]
            g = ifgo[:, self.hidden_size * 2 : self.hidden_size * 3]
            o = ifgo[:, self.hidden_size * 3 : self.hidden_size * 4]

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c_n[0] = f * c_n[0] + i * g
            h_n[0] = o * torch.tanh(c_n[0])

            # layer 1
            for layer_id in range(1, self.num_layers):
                ifgo = (
                    torch.matmul(h_n[layer_id - 1], self.wi[layer_id])
                    + torch.matmul(h_n[layer_id], self.wh[layer_id])
                    + self.bias[layer_id]
                )

                i = ifgo[:, 0 : self.hidden_size]
                f = ifgo[:, self.hidden_size : self.hidden_size * 2]
                g = ifgo[:, self.hidden_size * 2 : self.hidden_size * 3]
                o = ifgo[:, self.hidden_size * 3 : self.hidden_size * 4]

                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)

                c_n[layer_id] = f * c_n[layer_id] + i * g
                h_n[layer_id] = o * torch.tanh(c_n[layer_id])

            out[t] = h_n[self.num_layers - 1].clone()

        return out, h_n, c_n
