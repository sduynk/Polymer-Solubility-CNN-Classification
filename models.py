from torchvision.models import (
    resnet50,
    efficientnet_b0,
    convnext_tiny
)
import torch
import torch.nn as nn
import torch.nn.functional as F

### functions to get models and optionally load weights

def resnet(state_dict_path=None):
    model = resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 4)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def efficientnet(state_dict_path=None):
    model = efficientnet_b0(weights='DEFAULT')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def convnext(state_dict_path=None):
    model = convnext_tiny(weights='DEFAULT')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model
