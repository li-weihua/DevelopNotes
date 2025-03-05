import torch
import mma

torch.manual_seed(1)

device = torch.device("cuda")

A = torch.randint(low = -128, high = 127, size=(8, 16), device=device).to(torch.int8).as_strided(size=(8,16), stride=(16,1))
B = torch.randint(low = -128, high = 127, size=(16, 8), device=device).to(torch.int8).as_strided(size=(16,8), stride=(1,16))

# reference
D0 = torch.matmul(A.float(), B.float())
D = mma.matmul_8x8x16_s8(A, B)

print(f"max difference: {(D-D0).abs().max().item()}")
