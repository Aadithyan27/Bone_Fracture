import torch

x = torch.randn(10000, 10000, device="cuda")
y = torch.mm(x, x)
print("Done on:", y.device)
