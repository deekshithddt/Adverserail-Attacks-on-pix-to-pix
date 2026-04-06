import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def to_model_tensor(img_gray, image_size, device):
    """Convert PIL image to model input tensor."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img_gray).unsqueeze(0).to(device)

def pixel_eps_to_model_eps(pixel_eps):
    """Convert pixel epsilon [0,1] back to model scale [-1, 1]."""
    return pixel_eps * 2.0

def denorm_to_uint8(tensor):
    """Convert model output tensor [-1, 1] back to uint8 array [0, 255]."""
    out = tensor.clone().detach().squeeze().cpu().numpy()
    out = (out + 1.0) / 2.0
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

def fgsm_attack(model, x, epsilon):
    """Fast Gradient Sign Method (FGSM) to maximize L1 difference on output."""
    y_clean = model(x).detach()
    # Add a tiny noise jitter because if x_adv == x exactly, the distance loss is 0 and the mathematical gradient is 0 at the origin!
    jitter = torch.empty_like(x).uniform_(-1e-4, 1e-4)
    x_adv = (x.clone().detach() + jitter).requires_grad_(True)
    
    y_adv = model(x_adv)
    loss = nn.L1Loss()(y_adv, y_clean)
    
    model.zero_grad()
    loss.backward()
    
    perturbation = epsilon * x_adv.grad.detach().sign()
    x_adv = x_adv + perturbation
    x_adv = torch.clamp(x_adv, -1.0, 1.0).detach()
    
    y_adv_final = model(x_adv).detach()
    final_loss = nn.L1Loss()(y_adv_final, y_clean).item()
    return y_clean, x_adv, y_adv_final, final_loss

def pgd_attack(model, x, epsilon, steps, alpha, random_start, project):
    """Projected Gradient Descent (PGD) attack for generative model output."""
    y_clean = model(x).detach()
    x_adv = x.clone().detach()
    
    if alpha is None:
        alpha = epsilon / max(steps, 1)

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
        
    for _ in range(steps):
        x_adv.requires_grad_(True)
        y_adv = model(x_adv)
        loss = nn.L1Loss()(y_adv, y_clean)
        
        model.zero_grad()
        loss.backward()
        
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + alpha * grad_sign
        
        if project:
            eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + eta, -1.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
        
        x_adv = x_adv.detach()
            
    y_adv_final = model(x_adv).detach()
    final_loss = nn.L1Loss()(y_adv_final, y_clean).item()
    return y_clean, x_adv, y_adv_final, final_loss

def deepfool_attack(model, x, steps, shift_threshold, overshoot):
    """Approximate DeepFool for image-to-image by doing iterative gradient ascent 
    until a difference threshold is reached."""
    y_clean = model(x).detach()
    jitter = torch.empty_like(x).uniform_(-1e-4, 1e-4)
    x_adv = x.clone().detach() + jitter
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        y_adv = model(x_adv)
        loss = nn.L1Loss()(y_adv, y_clean)
        
        if loss.item() >= shift_threshold:
            break
            
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.detach()
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
        
        # Ascent step based on L2 norm of gradient
        x_adv = x_adv + overshoot * grad / (grad_norm + 1e-8)
        x_adv = torch.clamp(x_adv, -1.0, 1.0).detach()
        
    y_adv_final = model(x_adv).detach()
    final_loss = nn.L1Loss()(y_adv_final, y_clean).item()
    return y_clean, x_adv, y_adv_final, final_loss

def cw_attack(model, x, epsilon, steps, alpha, c):
    """Carlini-Wagner inspired untargeted attack to maximize L1 output difference 
    while minimizing L2 perturbation penalty via tanh optimization."""
    y_clean = model(x).detach()
    
    jitter = torch.empty_like(x).uniform_(-1e-3, 1e-3)
    w = (torch.atanh(x.clone().detach() * 0.9999) + jitter).requires_grad_(True)
    
    if alpha is None:
        alpha = 0.01

    optimizer = torch.optim.Adam([w], lr=alpha)
    
    for _ in range(steps):
        x_adv = torch.tanh(w)
        y_adv = model(x_adv)
        
        # Penalty for distorting the input
        loss_dist = nn.MSELoss()(x_adv, x)
        
        # Adversarial objective: maximize L1 difference from clean generated image
        loss_adv = -nn.L1Loss()(y_adv, y_clean)
        
        loss = loss_dist + c * loss_adv
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    x_adv_final = torch.tanh(w).detach()
    y_adv_final = model(x_adv_final).detach()
    final_loss = nn.L1Loss()(y_adv_final, y_clean).item()
    return y_clean, x_adv_final, y_adv_final, final_loss
