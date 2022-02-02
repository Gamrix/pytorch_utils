# %%
import sys

import torch
from torch import complex32, float64, int32, int64, nn


def setUp():
    global prev_symbolic_shapes_test_enabled
    prev_symbolic_shapes_test_enabled = (
        torch._C._jit_symbolic_shapes_test_mode_enabled()
    )
    torch._C._jit_set_symbolic_shapes_test_mode(True)


def tearDown():
    torch._C._jit_set_symbolic_shapes_test_mode(prev_symbolic_shapes_test_enabled)


def prop_dtype_on_graph(graph, input_shapes, input_dtypes):
    graph_inputs = list(graph.inputs())
    for input, shape, dtype in zip(graph_inputs, input_shapes, input_dtypes):
        input.setType(dtype.with_scalar_type(dtype).with_sizes(shape))
    torch._C._jit_pass_propagate_shapes_on_graph(graph)
    torch._C._jit_pass_propagate_dtype(graph)


setUp()


@torch.jit.script
def foo(x, y):
    return x + y


graph = foo.graph

graph_inputs = list(graph.inputs())
print(repr(graph_inputs[0].type()))
print(repr(graph_inputs[1].type()))
print(torch.rand(1, 2, 3, dtype=torch.float32).type())

int_tensor = torch.ones(1, dtype=torch.int)
long_zerodim = torch.tensor(1, dtype=torch.long)
print(repr(int_tensor.type()))
print(isinstance(graph_inputs[0].type(), torch.TensorType))
input_shape_objs = [int_tensor, long_zerodim]
# help(graph_inputs[0].type())
print(graph_inputs[0])
# help(graph_inputs[0])

dtype = torch.int
shape = [1, 1]
graph_in = graph_inputs[0]
#print(graph_in.type().dtype())
# input.setType(input.type().with_scalarType(dtype).with_sizes(shape))

"""
for input, shape_obj in zip(graph_inputs, input_shape_objs):
    input.setType(input.type().with_sizes(shape_obj.shape).with_dtype(shape_obj.dtype))
"""


# torch._C._jit_pass_propagate_shapes_on_graph(graph)
# torch._C._jit_pass_propagate_dtype(graph)
# prop_dtype_on_graph(foo.graph, [torch.randn(2, 3), torch.randn(2, 3)], [torch.float32, torch.float32])

# %%

@torch.jit.script
def hardswish(x):
    return torch.nn.Hardswish(x)

print(hardswish.graph())


# %%
print(graph_in.type().dtype())
x = graph_in.type().with_dtype(torch.long).with_sizes((1, 1))
print(x.scalarType())
print(x.dtype())

# Just making sure that we need complicated type promotion logic


print((int_tensor + long_zerodim).dtype)
# Note how int was not promoted to long

# %%

type_0 = graph_inputs[0].type().with_dtype(torch.long).with_sizes((2, 2))
graph_inputs[0].setType(type_0)
type_1 = graph_inputs[1].type().with_dtype(torch.int).with_sizes((2, 2))
graph_inputs[1].setType(type_1)

torch._C._jit_pass_propagate_shapes_on_graph(graph)
torch._C._jit_pass_propagate_dtype(graph)

try:
    del output_dtype
except NameError:
    pass

print(f"before refcount: {sys.getrefcount(torch.long)}")
output_dtype = graph.findNode("aten::add").output().type().dtype()
print(f"after refcount 2: {sys.getrefcount(output_dtype)}")
print(f"after refcount: {sys.getrefcount(torch.long)}")
output_dtype2 = list(graph.outputs())[0].type().dtype()
print(output_dtype)
print(output_dtype == torch.long)
print(output_dtype is torch.long)
print(output_dtype2 is torch.long)

# %%
print(list(graph.outputs())[0].type().dtype())

print(torch.rand((1, 1), dtype=torch.int32))

# %%
# Test Eager Execution


def get_rand_tensor(shape, dtype):
    if dtype in (int32, int64):
        rand_tensor = torch.randint(0, 10, shape, dtype=dtype)
    else:
        rand_tensor = torch.rand(shape, dtype=dtype)
    return rand_tensor


x, y = get_rand_tensor([2, 2], int32), get_rand_tensor([2, 2], int64)
eager_res = x + y
print(eager_res.dtype)

# %%
# Test out scalar execution
# Ok, note that scalars need to be annotated.
@torch.jit.script
def mul(x: float, y: int):
    return x * y


z = mul(2.1, 3)
print(z)

mul_graph = mul.graph
print(mul_graph)
graph_inputs = list(graph.inputs())
print(repr(graph_inputs[0].type()))
print(repr(graph_inputs[1].type()))

# %%
print(float == torch.float32)
print(float == torch.float64)
# %%

# Debugging some tests:

from typing import Tuple


@torch.jit.script
def adaptive_avg_pool2d_fn(input, output_size: Tuple[int]):
    return torch.nn.functional.adaptive_avg_pool2d(input, output_size)

# adaptive = torch.jit.freeze(adaptive_avg_pool2d_fn)
print(adaptive_avg_pool2d_fn.graph)

# %%
from pprint import pprint

pprint(torch._C._nn.__dict__.keys())

# %%
# Playing around with tensors with a dim of size 0

x = torch.randn(2, 0, 3)
y = torch.randn(1, 3)

print(x)
print(x + y)

print(help(x))

# %%
from torch import TensorType
help(TensorType)

# %%
a = TensorType.create_from_tensor(x)
b = TensorType.create_from_tensor(x)
print(id(a))
print(id(b))
# print(hash(TensorType.create_from_tensor(x))

# %%
