import torch
import copy

def deep_clone_tensor(x):
    return x.detach().clone()

def deep_clone_model(model):
    return copy.deepcopy(model)

def normalize_torch(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def tanh_torch(x):
    return torch.tanh(x)