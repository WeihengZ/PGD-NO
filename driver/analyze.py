import torch
import numpy as np
import argparse
import pickle
from dataloader import create_data_loaders
from models.transolver_model import Model as Transolver_Model
from models.Transolver_seg import Model as Transolver_SEG_Model
import os

def analyze_attention(model_name, model_path, data_path, sample_idx, predicted_feature_name="pressure"):
    """
    Analyze attention scores for a specific sample.
    
    Args:
        model_name: Name of the model ('transolver_seg')
        model_path: Path to the trained model
        data_path: Path to the data directory
        sample_idx: Index of the sample to analyze
        predicted_feature_name: Name of the predicted feature
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader for the specific sample
    print(f"\nLoading sample {sample_idx}...")
    _, _, test_loader, _ = create_data_loaders(
        data_path,
        batch_size=1,
        train_index=[],
        val_index=[],
        test_index=[sample_idx],
        shuffle=False,
        predicted_feature_name=predicted_feature_name
    )
    
    # Get the sample
    for batch_data in test_loader:
        coorf, seg_matrix, target, sim_ids = batch_data
        coorf = coorf.to(device)
        seg_matrix = seg_matrix.to(device)
        sim_id = sim_ids[0]
        print(f"Loaded sample: {sim_id}")
        print(f"Input shape: {coorf.shape}")
        print(f"Number of tokens: {coorf.shape[1]}")
        if model_name in ['transolver_seg', 'transolver_seg_v2']:
            print(f"Seg matrix shape: {seg_matrix.shape}")
        break
    
    # Create and load model
    print(f"\nLoading model from {model_path}...")
    if model_name == 'transolver':
        model = Transolver_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act='gelu',
            mlp_ratio=2,
            slice_num=64
        )
    elif model_name == 'transolver_seg':
        model = Transolver_SEG_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act='gelu',
            mlp_ratio=2,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = model.float().to(device)
    model.eval()
    
    # Extract attention scores
    print(f"\nExtracting attention scores...")
    if model_name == 'transolver_seg':
        attention_results = model.extract_attention_scores((coorf, seg_matrix))
    else:
        raise ValueError(f"Model name {model_name} not supported for attention extraction")
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Attention Analysis Results")
    print(f"{'='*80}")
    print(f"Sample ID: {sim_id}")
    print(f"Attention shape: (B, H, N_query_nodes, N_tokens)")
    print(f"Total Query Nodes: {coorf.shape[1]}")
    print(f"Total Slice Tokens: {seg_matrix.shape[1]}")
    print(f"Number of Layers: {len(attention_results['attention_scores'])}")
    
    # Display attention scores for each layer
    print(f"\n{'='*80}")
    print(f"Attention Scores (from coordinates to slice tokens)")
    print(f"{'='*80}")
    
    for layer_idx, attn_scores in enumerate(attention_results['attention_scores']):
        # attn_scores is (B, H, N, S) for transolver_seg
        B, H, N, S = attn_scores.shape
        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Shape: {attn_scores.shape} (B={B}, H={H}, N_query_nodes={N}, N_tokens={S})")
        print(f"  Min: {attn_scores.min():.6f}, Max: {attn_scores.max():.6f}, Mean: {attn_scores.mean():.6f}")
        print(f"  Std: {attn_scores.std():.6f}")
        
        # Show overall statistics per head
        print(f"  Per-head statistics (averaged over all query nodes):")
        for head_idx in range(H):
            head_attn = attn_scores[0, head_idx, :, :]  # (N, S)
            head_mean = head_attn.mean()
            head_max = head_attn.max()
            head_max_query, head_max_token = np.unravel_index(head_attn.argmax(), head_attn.shape)
            print(f"    Head {head_idx}: mean={head_mean:.6f}, max={head_max:.6f} "
                  f"(query_node={head_max_query}, token={head_max_token})")
    
    return attention_results


def main():
    parser = argparse.ArgumentParser(description='Analyze attention scores for a specific sample and token')
    parser.add_argument('--model_name', type=str, default='transolver_seg', 
                       choices=['transolver', 'transolver_seg'], help='Model name')
    parser.add_argument('--model_path', type=str, 
                       default='trained_models/best_model_transolver_seg_pressure.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                       default='/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/normalized/',
                       help='Path to data directory')
    parser.add_argument('--sample_idx', type=int, required=True, help='Sample index to analyze')
    parser.add_argument('--predicted_feature_name', type=str, default='pressure',
                       help='Name of predicted feature')
    
    args = parser.parse_args()
    
    analyze_attention(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        sample_idx=args.sample_idx,
        predicted_feature_name=args.predicted_feature_name
    )


if __name__ == "__main__":
    main()

