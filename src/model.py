import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, UNFROZEN_LAYERS

def build_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #Disables parameter updating in early layers
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in UNFROZEN_LAYERS):
            param.requires_grad = True
        else:
            param.requires_grad = False
    #Replace head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model