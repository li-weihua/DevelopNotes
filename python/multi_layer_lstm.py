import torch
import torch.nn as nn

torch.set_grad_enabled(False)

torch.manual_seed(1)

seq_len = 22
batch_size = 7
input_size = 10
hidden_size = 20
num_layers = 5

x = torch.randn(seq_len, batch_size, input_size)

rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=False)

y0, (h0, c0) = rnn.forward(x)


wi = []
wh = []
bias = []

st = rnn.state_dict()

for layer_id in range(num_layers):
    wi.append(st[f"weight_ih_l{layer_id}"].T)
    wh.append(st[f"weight_hh_l{layer_id}"].T)
    bias.append(st[f"bias_ih_l{layer_id}"] + st[f"bias_hh_l{layer_id}"])

y1 = torch.zeros(seq_len, batch_size, hidden_size)
hn = torch.zeros(num_layers, batch_size, hidden_size)
cn = torch.zeros(num_layers, batch_size, hidden_size)

for t in range(seq_len):
    # layer 0
    ifgo = torch.matmul(x[t], wi[0]) + torch.matmul(hn[0], wh[0]) + bias[0]

    i = ifgo[:, 0: hidden_size]
    f = ifgo[:, hidden_size : hidden_size * 2]
    g = ifgo[:, hidden_size * 2 : hidden_size * 3]
    o = ifgo[:, hidden_size * 3 : hidden_size * 4]

    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)

    cn[0] = f * cn[0] + i * g
    hn[0] = o * torch.tanh(cn[0])

    # layer 1
    for layer_id in range(1, num_layers):
        ifgo = torch.matmul(hn[layer_id-1], wi[layer_id]) + torch.matmul(hn[layer_id], wh[layer_id]) + bias[layer_id]

        i = ifgo[:, 0: hidden_size]
        f = ifgo[:, hidden_size : hidden_size * 2]
        g = ifgo[:, hidden_size * 2 : hidden_size * 3]
        o = ifgo[:, hidden_size * 3 : hidden_size * 4]

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        cn[layer_id] = f * cn[layer_id] + i * g
        hn[layer_id] = o * torch.tanh(cn[layer_id])


    y1[t] = hn[num_layers-1].clone()


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
