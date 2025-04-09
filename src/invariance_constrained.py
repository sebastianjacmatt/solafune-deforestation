import random
import torch
import torch.nn.functional as F
import numpy as np

def independent_mh_sampler(model, G, x, y, n_steps):

    g_t = np.random.choice(G)  # Initial state
    gx_t = apply_albumentations(g_t, x)
    loss_t = model.dice_loss_fn(model(gx_t.unsqueeze(0)), y.unsqueeze(0)) + model.bce_loss_fn(model(gx_t.unsqueeze(0)), y.unsqueeze(0))
    samples = [(g_t, loss_t.item())]
    
    for _ in range(n_steps):
        g_prop = random.choice(G)
        gx_prop = apply_albumentations(g_prop,x)
        loss_prop = model.dice_loss_fn(model(gx_prop.unsqueeze(0)), y.unsqueeze(0)) + model.bce_loss_fn(model(gx_prop.unsqueeze(0)), y.unsqueeze(0))

        acceptance_ratio = min(1.0, loss_prop.item() / loss_t.item()) if loss_t.item() > 0 else 1.0   
        if np.random.rand() < acceptance_ratio:
            g_t, loss_t = g_prop, loss_prop
        
        samples.append((g_t, loss_t.item()))
    
    return samples

def primal_dual_augmentation(model, data_batch, G, optimizer, gamma, epsilon=0.01,
                             eta_p=0.001, eta_d=0.001, n_mh_steps=10, m_samples=5, device='cuda'):
    batch_size = len(data_batch)
    transformed_losses = []

    for x, y in data_batch:
        x, y = x.to(device), y.to(device)
        mh_samples = independent_mh_sampler(model, G, x, y, n_steps= n_mh_steps)
        selected = random.sample(mh_samples, k=min(m_samples, len(mh_samples)))

        losses = []
        for g, _ in selected:
            gx = apply_albumentations(g, x)
            pred = model(gx.unsqueeze(0))
            losses.append(
                model.dice_loss_fn(pred, y.unsqueeze(0)) +
                model.bce_loss_fn(pred, y.unsqueeze(0))
            )
        transformed_losses.append(sum(losses) / len(losses))

    # Augmented (robust) loss
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
    optimizer.step()

    # Dual update
    gamma = max(0, gamma + eta_d * slack.item())

    return L_total.item(), gamma

def apply_albumentations(g, x):
    x_np = x.permute(1, 2, 0).cpu().numpy()
    x_aug = g(image=x_np)["image"]
    return torch.from_numpy(x_aug).permute(2, 0, 1)
