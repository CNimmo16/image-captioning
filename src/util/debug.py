import torch
            
def compute_grad_norms(model: torch.nn.Module):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

def detect_gradient_issues(grad_norms, vanish_thresh: int, explode_thresh: int):
    vanishing = {name: norm for name, norm in grad_norms.items() if norm < vanish_thresh}
    exploding = {name: norm for name, norm in grad_norms.items() if norm > explode_thresh}

    if vanishing:
        print(f'❗Vanishing gradients detected: {vanishing}')
    if exploding:
        print(f'❗Exploding gradients detected: {exploding}')

    return vanishing, exploding
