import torch

def assert_shape(tensor: torch.Tensor, shape):
    assert len(tensor.shape) == len(shape), f"Expected {len(shape)} dimensions, got {len(tensor.shape)}"
    
    for i in range(len(tensor.shape)):
        if shape[i] != "*" and tensor.shape[i] != shape[i]:
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
