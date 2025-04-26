from itertools import product
import torch
import os
import sys
from src.utils.train_utils import prepare_dataloaders, random_crop_icl
from src.model import Model
from src.invariance_constrained import invariance_constrained_fit
from src.config import GAMMA, EPSILON, ETA_P, ETA_D, N_MH_STEPS, LEARNING_RATE_OPT

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

# Best Configuration:
#{'learning_rate': 0.001, 'gamma': 0.5, 'epsilon': 0.05, 'eta_p': 0.01, 'eta_d': 0.01}
def hyperparameter_tuning():
    """
    Perform hyperparameter tuning for invariance_constrained_fit.
    Updates the relevant values from config.py globally
    """
    # Define hyperparameter search space
    gamma_values = [0.1, 0.5]
    epsilon_values = [0.01, 0.05]
    eta_p_values = [0.001, 0.01]
    eta_d_values = [0.001, 0.01]
    n_mh_steps = [2]

    # Store the best configuration and its validation loss
    best_config = None
    best_val_loss = float("inf")

    # Iterate over all combinations of hyperparameters
    for gamma, epsilon, eta_p, eta_d, n_mh in product(
     gamma_values, epsilon_values, eta_p_values, eta_d_values, n_mh_steps
    ):
        print(f"Testing configuration: gamma={gamma}, epsilon={epsilon}, eta_p={eta_p}, eta_d={eta_d}, n_mh={n_mh}")

        # Update global hyperparameters
        global GAMMA, EPSILON, ETA_P, ETA_D
        GAMMA = gamma
        EPSILON = epsilon
        ETA_P = eta_p
        ETA_D = eta_d
        N_MH_STEPS = n_mh

        # Prepare data loaders
        train_loader, val_loader = prepare_dataloaders(random_crop_icl())

        # Initialize model, optimizer, and scheduler
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_OPT)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train the model
        trained_model = invariance_constrained_fit(
            model, train_loader, val_loader, optimizer, scheduler, num_epochs=10, device="cuda"
        )

        # Evaluate validation loss
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to("cuda")
                masks = batch["mask"].to("cuda")
                logits = trained_model(images)
                loss = model.dice_loss_fn(logits, masks) + model.bce_loss_fn(logits, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}")

        # Update the best configuration if the current one is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = {
                "gamma": gamma,
                "epsilon": epsilon,
                "eta_p": eta_p,
                "eta_d": eta_d,
            }

    print("\nBest Configuration:")
    print(best_config)
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    hyperparameter_tuning()