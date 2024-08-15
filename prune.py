#https://www.youtube.com/watch?v=h3zwAOS5Ivg

import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.1):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

model = YOLO('yolov8s.pt')

results = model.val(data="coco8.yaml")
print(f"mAP50-95: {results.box.map}")

torch_model = model.model

print(torch_model)

print("Pruning model...")
pruned_torch_model = prune_model(torch_model, amount=0.1)
print("Model pruned.")
