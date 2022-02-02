#%%
import torchvision.models as models

import torch
from torch.jit._passes._property_propagation import apply_input_props_using_example

torch._C._jit_set_symbolic_shapes_test_mode(True)

m = torch.jit.script(models.mobilenet_v3_small())
m.eval()
f = torch.jit.freeze(m)
graph = f.graph

# print(graph)


# Need to figure out what the device of the inputs are.

# What does m.cuda() do?


# Set the inputs of the graph appropriately

# Currently erase shape pass isn't working correctly
# torch._C._jit_pass_erase_shape_information(graph)
input_val = torch.rand(1, 3, 224, 224, dtype=torch.float32)
apply_input_props_using_example(graph, [input_val])

# Do the graph propagation
print(graph)
# torch._C._jit_pass_propagate_shapes_on_graph(graph)
# torch._C._jit_pass_propagate_dtype(graph)




# %%

# Check the Graph Inputs
graph_in = list(graph.inputs())
print(graph_in[1].type().dtype())
# print(graph.features[1].block[0][0].weight_fused_bn)
all_nodes = list(graph.nodes())

# %%

for node in all_nodes:
    if node.kind() == "aten::relu_":
        relu_node = node
        break

print(relu_node)
relu_input = list(relu_node.inputs())[0]
relu_output = relu_node.output()
print(relu_input.type().sizes())
print(relu_output.type().sizes())
# %input.40 : Tensor = aten::mul(%121, %input.34)
# %%
broken_node = None

for n in all_nodes:
    # if n.kind() == "aten::conv2d":
    #     break
    if len(list(n.outputs())) == 1 and isinstance(n.output().type(), torch._C.TensorType):
        if n.output().type().dtype() == None:
            broken_node = n
            break

print(broken_node)
node_out = broken_node.output()
print(node_out.type().sizes())
print(node_out.type().dtype())
node_in = list(broken_node.inputs())
print(node_in[0].type().dtype())
node_in_1 = node_in[1]
print(node_in_1.type().dtype())
print(node_in_1.type().sizes())
# print(list(node_in_1.node().inputs())[0])

# %%

graph_out = list(graph.outputs())
assert len(graph_out) == 1
print(graph_out[0].type().dtype())
## %%
# Try out Quantized Mobilenet V3
# mobilenet_v3_large = models.quantization.mobilenet_v3_large()

"""
model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()
# run the model with quantized inputs and weights
out = model(torch.rand(1, 3, 224, 224))
"""

# Figure out what the type of the first input into the graph is:
graph_in = list(graph.inputs())
in_0 = graph_in[0]
print(graph_in[0].type())
print(isinstance(in_0.type(), torch._C.ClassType))
print(in_0.debugName() == "self")
# %%
help(graph)

# %%
