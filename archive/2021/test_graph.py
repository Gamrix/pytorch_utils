#%%
import torch
from torch import nn



class M(nn.Module):
    def __init__(self):
        self.a = 1
        super(M, self).__init__()

    def forward(self, x):
        return str(self.a)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.a = torch.tensor([1.1])
        self.b = torch.tensor([2.2])

    def forward(self, x):
        return self.a + self.b

m = torch.jit.script(Module())

print(m.graph)
m.eval()
y = torch.jit.freeze(m)
print(m.graph)

# %%
