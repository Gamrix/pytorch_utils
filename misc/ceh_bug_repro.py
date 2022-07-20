# %%
import torch
from typing import Dict

def fn(x, y, cond: bool, d: Dict[str, torch.Tensor]):
    if cond:
        m = x.relu()
        f1 = torch.rand((2, 2))
        d["test"] = f1
        z = d["test"]
    else:
        m = y.gelu()
        f2 = torch.rand((3, 2))
        d["test"] = f2
        z = d["test"]
    return m, z

fn_s = torch.jit.script(fn)
x = torch.rand((2, 2))
y = torch.rand((2, 2))
d = {"x": x}
print(fn_s.graph)
fn_s(x, y, 1, d)

#%%
