#%%
import torch

@torch.jit.script
def fn(x):
    torch.nonzero(x)

print(fn.graph)

x = torch.zeros(2,3)
y = torch.nonzero(x, as_tuple=True)
print(y)
x = torch.rand(2,3)
print(tuple(torch.nonzero(x).size()))
# %%
