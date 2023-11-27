import torch
from torchvision import transforms

transform_rev = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1. / 0.229, 1. / 0.224, 1. / 0.225])

def get_revd_perceptual(inputs, recons, perceptual_model):
    return torch.mean(perceptual_model(transform_rev(inputs.contiguous()), transform_rev(recons.contiguous())))
