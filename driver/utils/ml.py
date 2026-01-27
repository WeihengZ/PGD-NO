import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from utils.metric import compute_relative_error
from utils.metric import denormalize_pressure

def train(model_name, model, train_loader, val_loader, normalization_scalars, 
          num_epochs=100, learning_rate=0.0001, eval_freq = 10,
          save_path="models/best_model.pth", predicted_feature_name="pressure"):
    """
    Train the model with validation monitoring.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        normalization_scalars: Normalization scalars for denormalization
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save the best model
        
    Returns:
        dict: Training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=(len(train_loader) // batch_size + 1) * num_epochs,
        final_div_factor=1000.,
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.7, patience=20)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_relative_error': [],
        'best_val_error': float('inf')
    }
    
    print(f"Starting training for {num_epochs} epochs (model_name='{model_name}', assuming 'transolver' and coorf-only input)...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        for batch_data in train_loader:
            # New dataloader format: (coorf, seg_matrix, p, sim_id)
            coorf, seg_matrix, target, sim_ids = batch_data
            coorf, target = coorf.to(device), target.to(device)
            seg_matrix = seg_matrix.to(device)

            optimizer.zero_grad()
            if model_name == 'transolver':
                outputs = model(coorf)
            elif model_name == 'transolver_seg' or model_name == 'transolver_seg_v2':
                outputs = model((coorf, seg_matrix))
            else:
                raise ValueError(f"Model name {model_name} not supported")

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # lr_scheduler.step(loss.item())
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        if epoch % eval_freq == 0:
            model.eval()
            val_loss = 0.0
            val_relative_error = 0.0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    # New dataloader format: (coorf, seg_matrix, p, sim_id)
                    coorf, seg_matrix, target, sim_ids = batch_data
                    coorf, target = coorf.to(device), target.to(device)
                    seg_matrix = seg_matrix.to(device)

                    if model_name == 'transolver':
                        outputs = model(coorf)
                    elif model_name == 'transolver_seg' or model_name == 'transolver_seg_v2':
                        outputs = model((coorf, seg_matrix))
                    else:
                        raise ValueError(f"Model name {model_name} not supported")
                    
                    loss = criterion(outputs, target)
                    val_loss += loss.detach().cpu().item()
                    
                    # Compute relative error
                    relative_error = compute_relative_error(
                        outputs.detach().cpu().numpy(), 
                        target.detach().cpu().numpy(), normalization_scalars)
                    val_relative_error += relative_error
        
            val_loss /= len(val_loader)
            val_relative_error /= len(val_loader)
        
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_relative_error'].append(val_relative_error)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Relative Error: {val_relative_error:.6f}")
            
            # Save best model based on validation relative error
            if val_relative_error < history['best_val_error']:
                history['best_val_error'] = val_relative_error
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ New best model saved! (Relative Error: {val_relative_error:.6f})")
    
    print(f"Training completed! Best validation relative error: {history['best_val_error']:.6f}")
    return history

def test(model_name, model, test_loader, normalization_scalars, model_path="models/best_model.pth", predicted_feature_name="pressure"):
    """
    Test the trained model on test data.
    
    Args:
        model: The neural network model
        test_loader: Test data loader
        normalization_scalars: Normalization scalars for denormalization
        model_path: Path to the trained model
        
    Returns:
        dict: Test results
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
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    test_relative_error = 0.0
    all_predictions = {}
    all_targets = {}
    
    with torch.no_grad():
        # for batch_data in tqdm(test_loader, desc="Testing"):
        for batch_data in test_loader:
            # New dataloader format: (coorf, seg_matrix, p, sim_id)
            coorf, seg_matrix, target, sim_ids = batch_data
            coorf, target = coorf.to(device), target.to(device)
            seg_matrix = seg_matrix.to(device)

            if model_name == 'transolver':
                outputs = model(coorf)
            elif model_name == 'transolver_seg' or model_name == 'transolver_seg_v2':
                outputs = model((coorf, seg_matrix))
            else:
                raise ValueError(f"Model name {model_name} not supported")
            loss = criterion(outputs, target)
            test_loss += loss.item()
            
            # Compute field-level relative error
            relative_error = compute_relative_error(
                outputs.detach().cpu().numpy(), 
                target.detach().cpu().numpy(), normalization_scalars)
            test_relative_error += relative_error
            
            # Store predictions and targets for detailed analysis
            SIM_ID = sim_ids[0]
            all_predictions[SIM_ID] = outputs
            all_targets[SIM_ID] = target
    
    test_loss /= len(test_loader)
    test_relative_error /= len(test_loader)
    
    results = {
        'test_loss': test_loss,
        'test_relative_error': test_relative_error,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Relative Error: {test_relative_error:.6f}")
    
    return results