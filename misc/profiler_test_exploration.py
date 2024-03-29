#%%
import torch
import torch.nn as nn
from torch.profiler import (
    kineto_available,
    profile,
    record_function,
    supported_activities,
    DeviceType,
    ProfilerAction,
    ProfilerActivity,
    ExecutionGraphObserver,
    _utils,
)

# %%
x = torch.ones(10, 10)
y = torch.ones(1, 10)

with profile(with_stack=True, profile_memory=True) as p:
    _ = x + y

print(_.shape)


def find_add(nodes):
    for n in nodes:
        if n.name() == "aten::add":
            return n
        result = find_add(n.children)
        if result:
            return result


node = find_add(p.profiler.kineto_results.experimental_event_tree())
assert node is not None


"""
self.assertIsNotNone(node)

self.assertIsInstance(
    node.extra_fields,
    torch._C._autograd._ExtraFields_TorchOp)

self.assertIsInstance(
    node.parent.extra_fields,
    torch._C._autograd._ExtraFields_PyCCall)

self.assertEqual(node.children[0].name(), "aten::empty")
self.assertEqual(node.children[0].children[0].name(), "[memory]")
self.assertIsInstance(
    node.children[0].children[0].extra_fields,
    torch._C._autograd._ExtraFields_Allocation)
"""

node = find_add(p.profiler.kineto_results.experimental_event_tree())
print(node)
print(node.extra_fields.inputs)
print(node.extra_fields.inputs.scalars)
print(node.extra_fields.inputs.shapes)
print(node.extra_fields.inputs.dtypes_)

# %%

# Play with Sparse Tensors
import torch

i = [[0, 1, 1], [2, 0, 2]]
v = [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
s.stride()

# %%

input = torch.randn(1, 5, 10)
input = input.to_mkldnn()
print(input.layout, torch._mkldnn)
print(repr(input.layout))
try:
    input.stride()
except Exception as e:
    print(e)
# %%
