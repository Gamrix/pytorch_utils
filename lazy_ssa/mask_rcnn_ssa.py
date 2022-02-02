
#%%
# Pulled from
# https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/vision_maskrcnn/__init__.py

import torch
import torchvision


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
eval_model = model
model = torch.jit.script(model)
eval_model = torch.jit.script(model)
eval_model.eval()
eval_model = torch.jit.optimize_for_inference(eval_model)

print("Done")
# %%
