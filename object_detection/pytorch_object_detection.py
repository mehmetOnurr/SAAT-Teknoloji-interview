import torch
import cv2

import numpy as np
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

img = cv2.imread('images/ball3_sports_tv.jpg')
# Inference
results = model(img)

# Results
results.print()
# results.save()  # or .show()

print(results.xyxy[0] ) # img1 predictions (tensor)
print(results.pandas().xyxy[0] )