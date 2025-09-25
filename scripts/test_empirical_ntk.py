import importlib.util
import os
import torch
import torch.nn as nn

# import torchtangents.py from repository root via file path
repo_root = os.path.dirname(os.path.dirname(__file__))
tt_path = os.path.join(repo_root, 'torchtangents.py')
spec = importlib.util.spec_from_file_location('torchtangents', tt_path)
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
inputs = torch.randn(3, 4)
NTK = tt.empirical_ntk(model, inputs)
print('NTK shape:', NTK.shape)
print(NTK)
