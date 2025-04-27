import random
import torch
import torch.nn.functional as F
import numpy as np

def independent_mh_sampler(model, G, x, y, n_steps):
    """
    Independent Metropolings Hasting algorithm for invariance constrained learning.
    Algorithm 1 from paper https://proceedings.mlr.press/v202/hounie23a/hounie23a.pdf

    Args:
        model (torch.nn.Module): The segmentation model.
        G (list): A set of augmentations (transformations) to sample from.
        x (torch.Tensor): Input image tensor of shape (C, H, W).
        y (torch.Tensor): Ground truth mask tensor of shape (H, W).
        n_steps (int): Number of Metropolis-Hastings steps to perform.

    Returns:
        list: A list of tuples, where each tuple contains:
            - g (callable): The augmentation function.
            - loss (float): The loss value associated with the augmented input.    
    """

    g_t = np.random.choice(G) 
    gx_t, gy_t = apply_albumentations(g_t, x, y)
    loss_t = model.dice_loss_fn(model(gx_t.unsqueeze(0)), gy_t.unsqueeze(0)) + \
         model.bce_loss_fn(model(gx_t.unsqueeze(0)), gy_t.unsqueeze(0))
    samples = [(g_t, loss_t.item())]
    
    for _ in range(n_steps):
        g_prop = random.choice(G)
        gx_prop, gy_prop = apply_albumentations(g_prop, x, y)
        model_output = model(gx_prop.unsqueeze(0)) 
            
        loss_prop = model.dice_loss_fn(model_output, gy_prop.unsqueeze(0)) + \
                model.bce_loss_fn(model_output, gy_prop.unsqueeze(0))

        acceptance_ratio = min(1.0, loss_prop.item() / loss_t.item()) if loss_t.item() > 0 else 1.0   
        if np.random.rand() < acceptance_ratio:
            g_t, loss_t = g_prop, loss_prop
        
        samples.append((g_t, loss_t.item()))
    
    return samples

def primal_dual_augmentation(model, data_batch, G, optimizer, gamma, epsilon=0.01,
                             eta_p=0.001, eta_d=0.001, n_mh_steps=2, m_samples=1, device='cuda'):
    """ 
    Performs a single training step using the primal-dual optimization approach for invariance-constrained learning.
    Algorithm 2 from paper https://proceedings.mlr.press/v202/hounie23a/hounie23a.pdf

        Args:
        model (torch.nn.Module): The segmentation model to be trained.
        data_batch (list): A batch of data, where each element is a tuple (x, y):
            - x (torch.Tensor): Input image tensor of shape (C, H, W).
            - y (torch.Tensor): Ground truth mask tensor of shape (H, W).
        G (list): A set of augmentations (transformations) to sample from.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        gamma (float): Dual variable for the Lagrangian optimization.
        epsilon (float, optional): Constraint threshold for the augmented loss. Default is 0.01.
        eta_p (float, optional): Primal learning rate. Default is 0.001.
        eta_d (float, optional): Dual learning rate. Default is 0.001.
        n_mh_steps (int, optional): Number of Metropolis-Hastings steps for sampling augmentations. Default is 10.
        m_samples (int, optional): Number of sampled augmentations to use for the augmented loss. Default is 5.
        device (str, optional): Device to use for computation (e.g., "cuda" or "cpu"). Default is "cuda".

    Returns:
        tuple:
            - L_total.item() (float): The total Lagrangian loss for the batch.
            - gamma (float): The updated dual variable.
    """

    
    batch_size = len(data_batch)
    transformed_losses = []

    with torch.no_grad():

        for x, y in data_batch:
            x, y = x.to(device), y.to(device)
            mh_samples = independent_mh_sampler(model, G, x, y, n_steps = n_mh_steps)
            selected = random.sample(mh_samples, k=min(m_samples, len(mh_samples)))

            losses = []
            for g, _ in selected:
                gx, gy = apply_albumentations(g, x, y) 
                pred = model(gx.unsqueeze(0))
                losses.append(
                    model.dice_loss_fn(pred, gy.unsqueeze(0)) +
                    model.bce_loss_fn(pred, gy.unsqueeze(0))
                )
            transformed_losses.append(sum(losses) / len(losses))

    # Augmented loss
    lc = sum(transformed_losses) / batch_size
    slack = lc - epsilon

   # Clean loss
    l_clean = sum(
        model.dice_loss_fn(model(x.unsqueeze(0).to(device)), y.unsqueeze(0).to(device)) +
        model.bce_loss_fn(model(x.unsqueeze(0).to(device)), y.unsqueeze(0).to(device))
        for x, y in data_batch
    ) / batch_size

    # Lagrangian
    L_total = l_clean + gamma * slack

    # Backprop and update
    optimizer.zero_grad()
    L_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Dual update
    with torch.no_grad():
        gamma = max(0, gamma + eta_d * slack.item())

    return L_total.item(), gamma

def apply_albumentations(g, x, y):
    """
    Utility function for applying an Albumentations augmentation to a PyTorch tensor.'

    Args:
        g: Albumentations augmentation function.
        x (torch.Tensor): Input tensor of shape (C, H, W).
                y (torch.Tensor): Ground truth mask tensor of shape (C, H, W).

    Returns:
        torch.Tensor: Augmented tensor of shape (C, H, W).
    """
    
    x_np = x.permute(1, 2, 0).cpu().numpy()  # Convert image to (H, W, C) for Albumentations
    y_np = y.permute(1, 2, 0).cpu().numpy()
    augmented = g(image=x_np, mask=y_np)
    x_aug = augmented["image"]
    y_aug = augmented["mask"]
    return (
        torch.from_numpy(x_aug).permute(2, 0, 1).to(x.device),  # Convert back to (C, H, W)
        torch.from_numpy(y_aug).permute(2, 0, 1).to(y.device),  # Convert back to (H, W)
    )
