import os
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(device)

t1 = torch.randn(3, 3, device=device)
t2 = torch.randn(3, 3, device=device)
print(t1 + t2)
