# Usage: accelerate launch test_hf_accelerate.py

import os
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

import torch
from accelerate import Accelerator

accelerator = Accelerator()

device = accelerator.device
print(device)

t1 = torch.randn(3, 3, device=device)
t2 = torch.randn(3, 3, device=device)
print(t1 + t2)
