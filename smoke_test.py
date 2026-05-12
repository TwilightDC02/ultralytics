from ultralytics import YOLO
import torch

model = YOLO("ultralytics/cfg/models/ext/spdema-yolo26.yaml")
model.model.info()

x = torch.randn(1, 3, 640, 640)
y = model.model(x)
print(type(y))