# models/mobilevitv2_base.py
import torch.nn as nn
from torchvision.models import mobilenet_v2
def get_mobilevitv2_base(num_classes):
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
