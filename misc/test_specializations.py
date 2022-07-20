import torch

@torch.jit.script
def test_fuse(a, b):
    c = a * b
    d = c * b
    return d

print(test_fuse.graph)
