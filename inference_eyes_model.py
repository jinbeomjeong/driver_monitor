import torch
import torch.nn as nn
from torchvision import models as models


device = torch.device('cuda')
state_dict = torch.load('./models/eyes_model.pth', map_location=device)

num_classes = 2
eye_model = models.resnet18(weights=None)
eye_model.fc = nn.Linear(eye_model.fc.in_features, num_classes)
eye_model.to(device)
eye_model.load_state_dict(state_dict)

image_path_list = []


eye_model.eval()
with torch.no_grad():
    for image_path in image_path_list:
        print(image_path)
