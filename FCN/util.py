import numpy as np

def deNormImg(img):
    out = (img + 1) / 2
    return out.clip(0, 1)

def countParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def calReceptiveField(model):
    layers = []
    receptive_field = 1
    stride_product = 1
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layers.append([layer.stride[0], layer.kernel_size[0]])
        if isinstance(layer, nn.MaxPool2d):
            layers.append([2 , 2])
    for stride, kernel in layers[-1::-1]:
        stride_product *= stride
        receptive_field += (kernel -1) * stride_product
    return receptive_field
