import torch
import torch.nn as nn

torch.set_grad_enabled(False)

torch.manual_seed(1)

from multi_layer_lstm import MyLSTM

seq_len = 22
batch_size = 7
input_size = 10
hidden_size = 20
num_layers = 5


rnn = nn.LSTM(
    input_size, hidden_size, num_layers, batch_first=False, bidirectional=False
)

mylstm = MyLSTM(input_size, hidden_size, num_layers)
mylstm.set_weights(rnn.state_dict())


x = torch.randn(seq_len, batch_size, input_size)

y0, (h0, c0) = rnn.forward(x)
y1, hn, cn = mylstm.forward(x)

# check outputs
print(f"ref out : {y0.shape}, {y0.min()}, {y0.max()}")
print(f"cpp out : {y1.shape}, {y1.min()}, {y1.max()}")
print(f"diff    : {(y1-y0).abs().max()}")
print()

print(f"ref h0  : {h0.shape}, {h0.min()}, {h0.max()}")
print(f"cpp hn  : {hn.shape}, {hn.min()}, {hn.max()}")
print(f"diff    : {(hn-h0).abs().max()}")
print()

print(f"ref c0  : {c0.shape}, {c0.min()}, {c0.max()}")
print(f"cpp cn  : {cn.shape}, {cn.min()}, {cn.max()}")
print(f"diff    : {(cn-c0).abs().max()}")
print()
