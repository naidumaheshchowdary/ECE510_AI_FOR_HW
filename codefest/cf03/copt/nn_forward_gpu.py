# nn_forward_gpu.py
# ECE 410/510 HW4AI | Codefest 3 | COPT | Spring 2026
# Network: Linear(4->5) -> ReLU -> Linear(5->1), batch=16, on GPU

import sys
import torch
import torch.nn as nn

# Task 1: detect GPU, print device name, exit if not found
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print('ERROR: No CUDA GPU found. Requires a CUDA-capable GPU.')
    sys.exit(1)
print(f'Device      : {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
print(f'PyTorch     : {torch.__version__}')

# Task 2: define network and move to GPU
model = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
).to(device)
print(f'\nModel architecture:')
print(model)
print(f'Model device: {next(model.parameters()).device}')

# Task 3: random input [16,4], forward pass, print shape + device
torch.manual_seed(42)
x = torch.randn(16, 4).to(device)
output = model(x)
print(f'\nInput  shape : {list(x.shape)}')
print(f'Output shape : {list(output.shape)}')
print(f'Output device: {output.device}')

assert list(output.shape) == [16, 1], f'Expected [16,1], got {list(output.shape)}'
assert output.device.type == 'cuda', f'Output not on GPU: {output.device}'
print('\nAll checks passed. COPT complete.')
