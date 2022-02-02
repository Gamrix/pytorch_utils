#%%
import sys

import torch


# To requires a device arg,
@torch.jit.script
def f(x):
    return x.to("cpu")


print(f.graph)


@torch.jit.script
def f2(x):
    return "cpu"


print(f2.graph)


@torch.jit.script
def f(x):
    return x.to(None)


print(f.graph)


@torch.jit.script
def f(x):
    return x.to_mkldnn()


print(f.graph)


# %%


@torch.jit.script
def f(x):
    return x.cuda()

graph_in = next(f.graph.inputs()).type()

print(f.graph)
print(f.graph.findNode("aten::cuda").schema())
# %%
@torch.jit.script
def f(x, y: str):
    return x.to(y)


print(f.graph)
# %%
print()

# Figure out how to deal with if statements and loops


@torch.jit.script
def f(a, b, y: bool):
    z = a + 1
    if y:
        z = b
    return z


print(f.graph)


@torch.jit.script
def f(a, b, y: int):
    z = a
    for i in range(y):
        z = b + 1
    return z


print(f.graph)

# %%
@torch.jit.script
def f(a, b, y: int):
    c = y -1
    d = y -2
    for i in range(y):
        c = b + 1
        d = c + 1
    return c, d

print(f.graph)
# Remember that loops

# %%
