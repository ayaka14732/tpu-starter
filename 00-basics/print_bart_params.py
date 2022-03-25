import torch

bart = torch.hub.load('pytorch/fairseq', 'bart.base')

print(bart.model)
