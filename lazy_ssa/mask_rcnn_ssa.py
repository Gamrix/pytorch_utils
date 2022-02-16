#%%
# Pulled from
# https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/vision_maskrcnn/__init__.py

from collections import defaultdict
from pprint import pprint
from typing import List

import torch
import torchvision
import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input = torch.randn(3, 224, 224, device=device)

"""
# MaskRCNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
eval_model = model
model = torch.jit.script(model)
eval_model = torch.jit.script(model)
eval_model.eval()
eval_model = torch.jit.freeze(eval_model)
rcnn_input = [input]
eval_model(rcnn_input)
"""

# Resnet 50
model = torchvision.models.resnet50(pretrained=True)
eval_model = model
model = torch.jit.script(model)
eval_model = torch.jit.script(model)
eval_model.eval()
eval_model = torch.jit.freeze(eval_model)

graph = eval_model.graph
# eval_model(input)
# The fastest way to figure out how much faster things can be with JIT is to just measure the input vectors


start_time = time.perf_counter()
torch._C._jit_pass_propagate_shapes_on_graph(graph)
end_time = time.perf_counter()
print("Propagate shapes on graph: {}".format(end_time - start_time))


#%%
all_nodes = list(graph.nodes())
print("Number of Nodes: ", len(all_nodes))
node_counts = defaultdict(int)
for node in all_nodes:
    node_counts[node.kind()] += 1

x = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
pprint(x)


#%%


def get_shape_info_for_in(input):
    ivalue = input.toIValue()
    if isinstance(ivalue, torch.Tensor):
        return tuple(ivalue.size())
    if isinstance(ivalue, list):
        return tuple(ivalue)
    return ivalue


conv_node_shapes = defaultdict(int)
for node in all_nodes:
    # group all conv_nodes
    if node.kind() == "aten::conv2d":
        shape = tuple(get_shape_info_for_in(input) for input in node.inputs())
        conv_node_shapes[shape] += 1

pprint(conv_node_shapes)

# %%
inputs = list(node.inputs())
in0 = inputs[1]
print(type(in0.toIValue()))
# help(in0)
# print(list(in0.as_tensor().shape))
print(str(in0.type()))
print(str(in0.type()) == "Tensor")

# %%
