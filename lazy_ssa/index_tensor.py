#%%
import torch

# Understanding the index.tensor implementation

"""
    for i, index in enumerate(indices):
        if index is not None:
            check(
                index.dtype in [torch.long, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, byte or bool tensors",
            )
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero() # 2D tensor
                k = len(result)
                check(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                    IndexError,
                )
                for j in range(index.ndim):
                    check(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                        IndexError,
                    )
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
"""

x = torch.Tensor([[True, False, False], [False, True, True]])
nonzero = torch.nonzero(x)
print(nonzero)
# tensor([[0, 0],
#         [1, 2]])
print(nonzero.select(1, 0))
print(nonzero.select(1, 1))

# So basically for the boolean case, we are converting the boolean Tensor
# into a list of indicies for the relevant dimmensions

# This means that as index_tensor uses nonzero, it in turn can be a dynamic zone generator
# if the one of the indicies is not a Tensor sized object.

#%%
# Prove that aten::index is a dynamic zone generator
# when we are using a bool tensor to index.

x = torch.arange(20).view(4, 5)
print(x)

print("case Y")
y = x % 5 == 0
print(y)
print(x[y])  # This tensor has size [4]

print("case Z")
z = x % 2 == 0
print(z)
print(x[z])  # This tensor has size [10]


# %%
b = torch.arange(3)
a = x[b, b]
print(a)

# %% Testing how LazyTensor handles indexing
from typing import Sequence
import torch
import functools

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
)
import torch._lazy
import torch._lazy.config
import torch._lazy.metrics
import torch._lazy.ir_cache
import torch._lazy.ts_backend
import itertools
import yaml
import os
import pathlib
from unittest import skip

try:
    torch._lazy.ts_backend.init()
except RuntimeError:
    # Skip if already initialized
    pass


def test_index_bool():
    a = torch.rand([2, 2]).to(device="lazy")
    i = torch.tensor([[True, False], [False, True]]).to(device="lazy")
    c = a[i]
    print(torch._C._lazy._get_tensors_backend([c]))
    print(torch._C._lazy._get_tensors_text([c]))
    print(c.cpu())


def test_index_int():
    a = torch.rand([2, 2]).to(device="lazy")
    c = a[:, 1]
    print(torch._C._lazy._get_tensors_backend([c]))
    print(torch._C._lazy._get_tensors_text([c]))
    print(c.cpu())


def test_index_tensor():
    a = torch.rand([10, 10]).to(device="lazy")
    b = torch.Tensor([1, 4]).to(device="lazy", dtype=torch.long)
    c = a[b]
    print(torch._C._lazy._get_tensors_backend([c]))
    print(torch._C._lazy._get_tensors_text([c]))
    print(c.cpu())


test_index_tensor()

# %%
