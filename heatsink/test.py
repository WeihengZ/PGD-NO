import argparse
import os
import pickle

import torch

from dataloader import DATAPATH, load_data, create_data_loaders
from models.transolver_model import Model as Transolver_Model
from models.transolver_seg import Model as Transolver_Seg_Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
                       choices=["transolver", "transolver_seg"],
                       help="Model name: 'transolver' or 'transolver_seg'")
    args = parser.parse_args()

    model_name = args.model_name
    print(f"🧪 Testing {model_name} model on all heatsink data")
    print(f"  Data path: {DATAPATH}")

    # ------------------------------------------------------------------
    # Data - Load all data without splitting/shuffling
    # ------------------------------------------------------------------
    print("\n🔄 Loading all heatsink data...")
    data_dict = load_data(DATAPATH)
    num_samples = len(data_dict["vertices"])
    print(f"Found {num_samples} heatsink samples.")

    # Use all indices in order (no splitting, no shuffling)
    all_indices = list(range(num_samples))
    
    # Create a single data loader with all data, no shuffle
    train_loader, _, _ = create_data_loaders(
        data_dict,
        batch_size=1,
        train_index=all_indices,
        val_index=[],
        test_index=[],
        shuffle=False,  # Important: no shuffling
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n🏗️  Creating {model_name} model...")
    if model_name == "transolver":
        model = Transolver_Model(
            space_dim=3,   # vertices (x, y, z)
            out_dim=1,     # scalar temperature per vertex
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
            slice_num=64,
        )
    elif model_name == "transolver_seg":
        model = Transolver_Seg_Model(
            space_dim=3,   # vertices (x, y, z)
            out_dim=1,     # scalar temperature per vertex
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load pre-trained model
    save_path = "trained_models_heatsink/best_{}_model.pth".format(model_name)
    if os.path.exists(save_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"✅ Loaded model from {save_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {save_path}")

    # ------------------------------------------------------------------
    # Testing - Run inference on all samples
    # ------------------------------------------------------------------
    print("\n🔍 Running inference on all samples...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.float().to(device)
    model.eval()

    # Dictionary to store results: {sample_index: temperature_predictions_array}
    results = {}

    # Get the dataset to access indices
    dataset = train_loader.dataset

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(train_loader):
            vertices, temperatures, surface_nodes, seg_matrix = batch_data

            vertices = vertices.to(device)
            surface_nodes = surface_nodes.to(device)
            seg_matrix = seg_matrix.to(device)

            # Get the actual sample index from the dataset
            sample_index = dataset.indices[batch_idx]

            # Run inference
            if model_name == "transolver":
                outputs = model(vertices)  # (B, N, 1)
            elif model_name == "transolver_seg":
                outputs = model((surface_nodes, seg_matrix, vertices, None))  # (B, N, 1)
            else:
                raise ValueError(f"Model name {model_name} not supported")

            # Extract predictions for this batch (batch_size=1, so outputs[0] is the prediction)
            # Convert to numpy array and squeeze to 1D if needed
            pred_array = outputs[0].detach().cpu().numpy()
            if pred_array.ndim > 1:
                pred_array = pred_array.squeeze()
            
            # Store results with sample index as key
            results[sample_index] = pred_array

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_samples} samples...")

    print(f"✅ Completed inference on {len(results)} samples")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/heatsink/results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, "results_{}.pkl".format(model_name))
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"   Dictionary with {len(results)} samples")
    print(f"   Keys: sample indices (0 to {num_samples-1})")
    print(f"   Values: temperature prediction arrays")
    
    # Print a sample to verify
    if results:
        sample_key = list(results.keys())[0]
        print(f"\n   Example - Sample {sample_key}:")
        print(f"   Prediction shape: {results[sample_key].shape}")
        print(f"   Prediction range: [{results[sample_key].min():.4f}, {results[sample_key].max():.4f}]")

    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()

