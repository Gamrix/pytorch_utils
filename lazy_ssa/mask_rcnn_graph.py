#%%
# Pulled from
# https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/vision_maskrcnn/__init__.py

from collections import defaultdict
from pprint import pprint
from typing import List

import torch
import torchvision
import time

# MaskRCNN
device = "cpu"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
eval_model = model
model = torch.jit.script(model)
eval_model = torch.jit.script(model)
eval_model.eval()
eval_model = torch.jit.freeze(eval_model)
rcnn_input = [input]
graph = eval_model.graph

all_nodes = list(graph.nodes())
print("Number of Nodes: ", len(all_nodes))
node_counts = defaultdict(int)
for node in all_nodes:
    node_counts[node.kind()] += 1

x = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
pprint(x)

# %%


print("----MODEL GRAPH    ----")
print(graph)
print("----MODEL GRAPH END----")
