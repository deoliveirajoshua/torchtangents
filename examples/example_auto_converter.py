"""All-in-one conversion cheat sheet: make_tangential, lift, infinite_analog, inject."""
import torch
import torch.nn as nn
import torchtangents as tt

x1, x2 = torch.randn(4, 10), torch.randn(2, 10)
seq = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))

# 1. make_tangential: Convert nn.Sequential / standard layer into a tt.nn.Module
tt_seq = tt.convert.make_tangential(seq)
print("1. make_tangential NTK:", tt_seq.inf_kernel(x1, x2, get="ntk").shape)

# 2. lift: Trace arbitrary nn.Module (with branches/skip connections) into a tt.nn.LiftedModule
lifted = tt.convert.lift(seq)
print("2. lift NTK:           ", lifted.inf_kernel(x1, x2, get="ntk").shape)

# 3. infinite_analog: Extract a pure stateless kernel_fn from any nn.Module
kernel_fn = tt.convert.infinite_analog(seq)
print("3. infinite_analog NTK:", kernel_fn(x1, x2, get="ntk").shape)

# 4. inject: Mutate an existing nn.Module in-place to add .inf_kernel()
layer = nn.Linear(10, 2)
tt.convert.inject(layer)
print("4. inject NTK:         ", layer.inf_kernel(x1, x2, get="ntk").shape)
