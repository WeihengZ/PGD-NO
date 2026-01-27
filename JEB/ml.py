import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader_seperate import NORMALIZATION_PATH


def _load_normalization_stats(normalization_path: str = NORMALIZATION_PATH):
    """
    Load mean and std from normalization file.
    
    Returns:
        mean: Mean value for stress/volume_features
        std: Standard deviation for stress/volume_features
    """
    if not os.path.exists(normalization_path):
        print(f"Warning: Normalization file not found at {normalization_path}. Using default (mean=0, std=1)")
        return 0.0, 1.0
    
    data = np.load(normalization_path)
    # Try common keys for stress normalization
    if 'stress_mean' in data:
        mean = float(data['stress_mean'])
        std = float(data['stress_std'])
    elif 'mean' in data:
        mean = float(data['mean'])
        std = float(data['std'])
    elif 'volume_features_mean' in data:
        mean = float(data['volume_features_mean'])
        std = float(data['volume_features_std'])
    else:
        print(f"Warning: Could not find mean/std in {normalization_path}. Available keys: {list(data.keys())}")
        print("Using default (mean=0, std=1)")
        return 0.0, 1.0
    
    print(f"Loaded normalization stats: mean={mean:.6f}, std={std:.6f}")
    return mean, std


def _denormalize_stress(normalized_stress: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Denormalize stress values: denormalized = normalized * std + mean
    
    Args:
        normalized_stress: Normalized stress tensor
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized stress tensor
    """
    return normalized_stress * std + mean


def _relative_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Mean relative error |pred - target| / (|target| + eps) over all elements.
    """
    diff = torch.norm(pred - target, p=2)
    denom = torch.norm(target, p=2)
    rel = (diff / denom).item()
    return rel


def train(
    model,
    model_name,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    eval_freq: int = 10,
    save_path: str = "trained_models_JEB/best_model.pth",
) -> Dict:
    """
    Train Transolver on the JEB dataset.

    Batches from the dataloader have the form:
        vertices, stress, surface_nodes, seg_matrix
    and we currently use only `vertices` as input and `stress` as target.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 1
    total_steps = (len(train_loader) // batch_size + 1) * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        final_div_factor=1000.0,
    )

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load normalization statistics for denormalization
    stress_mean, stress_std = _load_normalization_stats()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_relative_error": [],
        "best_val_error": float("inf"),
    }

    print(f"Starting JEB {model_name} training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_data in train_loader:
            vertices, stress, surface_nodes, seg_matrix = batch_data

            vertices = vertices.to(device)  # (B, N, 3)
            surface_nodes = surface_nodes.to(device)  # (B, M, 6)
            seg_matrix = seg_matrix.to(device)  # (B, S, M)
            target = stress.to(device)      # (B, N) or (B, N, 1)

            # Ensure target has shape (B, N, 1) to match model output
            if target.ndim == 2:
                target = target.unsqueeze(-1)

            optimizer.zero_grad()
            if model_name == "transolver":
                outputs = model(vertices)       # (B, N, 1)
            elif model_name == "transolver_seg":
                outputs = model((surface_nodes, seg_matrix, vertices, None))       # (B, N, 1)
            else:
                raise ValueError(f"Model name {model_name} not supported")

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        if epoch % eval_freq == 0:
            model.eval()
            val_loss = 0.0
            val_relative_error = 0.0

            with torch.no_grad():
                for batch_data in val_loader:
                    vertices, stress, surface_nodes, seg_matrix = batch_data

                    vertices = vertices.to(device)
                    surface_nodes = surface_nodes.to(device)  # (B, M, 6)
                    seg_matrix = seg_matrix.to(device)  # (B, S, M)
                    target = stress.to(device)
                    if target.ndim == 2:
                        target = target.unsqueeze(-1)

                    if model_name == "transolver":
                        outputs = model(vertices)
                    elif model_name == "transolver_seg":
                        outputs = model((surface_nodes, seg_matrix, vertices, None))
                    else:
                        raise ValueError(f"Model name {model_name} not supported")
                    loss = criterion(outputs, target)
                    val_loss += loss.detach().cpu().item()

                    # Denormalize before computing relative error
                    outputs_denorm = _denormalize_stress(outputs, stress_mean, stress_std)
                    target_denorm = _denormalize_stress(target, stress_mean, stress_std)

                    rel_err = _relative_error(outputs_denorm, target_denorm)
                    val_relative_error += rel_err

            val_loss /= max(1, len(val_loader))
            val_relative_error /= max(1, len(val_loader))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_relative_error"].append(val_relative_error)

            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Relative Error: {val_relative_error:.6f}")

            if val_relative_error < history["best_val_error"]:
                history["best_val_error"] = val_relative_error
                torch.save(model.state_dict(), save_path)
                print(
                    f"  ✅ New best model saved! "
                    f"(Relative Error: {val_relative_error:.6f})"
                )

    print(
        f"Training completed! "
        f"Best validation relative error: {history['best_val_error']:.6f}"
    )
    return history


def test(
    model,
    model_name,
    test_loader,
    model_path: str = "trained_models_JEB/best_model.pth",
) -> Dict:
    """
    Test the trained model on JEB test data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")

    model = model.float().to(device)
    model.eval()

    # Load normalization statistics for denormalization
    stress_mean, stress_std = _load_normalization_stats()

    criterion = nn.MSELoss()
    test_loss = 0.0
    test_relative_error = 0.0
    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for batch_data in test_loader:
            vertices, stress, surface_nodes, seg_matrix = batch_data

            vertices = vertices.to(device)
            surface_nodes = surface_nodes.to(device)  # (B, M, 6)
            seg_matrix = seg_matrix.to(device)  # (B, S, M)
            target = stress.to(device)
            if target.ndim == 2:
                target = target.unsqueeze(-1)

            if model_name == "transolver":
                outputs = model(vertices)       # (B, N, 1)
            elif model_name == "transolver_seg":
                outputs = model((surface_nodes, seg_matrix, vertices, None))       # (B, N, 1)
            else:
                raise ValueError(f"Model name {model_name} not supported")

            loss = criterion(outputs, target)
            test_loss += loss.item()


            # Denormalize before computing relative error
            outputs_denorm = _denormalize_stress(outputs, stress_mean, stress_std)
            target_denorm = _denormalize_stress(target, stress_mean, stress_std)
            rel_err = _relative_error(outputs_denorm, target_denorm)
            test_relative_error += rel_err

            # For now, use simple running integer ids per batch element
            for b in range(outputs.shape[0]):
                idx = len(all_predictions)
                all_predictions[idx] = outputs[b].detach().cpu()
                all_targets[idx] = target[b].detach().cpu()

    test_loss /= max(1, len(test_loader))
    test_relative_error /= max(1, len(test_loader))

    results = {
        "test_loss": test_loss,
        "test_relative_error": test_relative_error,
        "predictions": all_predictions,
        "targets": all_targets,
    }

    print(f"Test Results (JEB {model_name}):")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Relative Error: {test_relative_error:.6f}")

    return results


