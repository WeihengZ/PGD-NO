import torch
import vtk
from vtk.util import numpy_support
import numpy as np
import os

def denormalize_pressure(normalized_pressure, normalization_scalars):
    """
    Denormalize pressure values back to original Cp scale.
    
    Args:
        normalized_pressure (torch.Tensor): Min-max normalized pressure values (Cp)
        normalization_scalars (dict): Normalization scalars
        
    Returns:
        torch.Tensor: Denormalized pressure values (Cp)
    """
    pressure_mean = normalization_scalars['pressure_mean']
    pressure_std = normalization_scalars['pressure_std']

    if isinstance(normalized_pressure, torch.Tensor):
        pressure_mean = torch.from_numpy(pressure_mean).to(normalized_pressure.device)
        pressure_std = torch.from_numpy(pressure_std).to(normalized_pressure.device)
    
    # Denormalize: normalized * range + min
    denormalized = normalized_pressure * pressure_std + pressure_mean
    
    return denormalized


def compute_relative_error(predicted, target, normalization_scalars):
    """
    Compute relative error in original scale.
    
    Args:
        predicted (torch.Tensor): Predicted pressure values (normalized)
        target (torch.Tensor): Target pressure values (normalized)
        normalization_scalars (dict): Normalization scalars
        
    Returns:
        float: Relative error
    """
    # Denormalize both predicted and target
    pred_denormalized = denormalize_pressure(predicted, normalization_scalars).squeeze()
    target_denormalized = denormalize_pressure(target, normalization_scalars).squeeze()
    
    # Compute relative error
    relative_error = np.linalg.norm(pred_denormalized - target_denormalized) / np.linalg.norm(target_denormalized)
    
    return relative_error


