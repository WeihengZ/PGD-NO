import os
import sys
import pickle
import numpy as np
import torch
import vtk
from vtk.util import numpy_support
from contextlib import contextmanager
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

# Import model classes
from models.transolver_model import Model as Transolver_Model
from models.transolver_seg import Model as Transolver_seg_Model
from dataloader_seperate import NORMALIZATION_PATH
from ml import _load_normalization_stats, _denormalize_stress

# Suppress VTK warnings
class VTKErrorFilter(vtk.vtkOutputWindow):
    def __init__(self):
        vtk.vtkOutputWindow.__init__(self)
        self.filtered_messages = ["Unsupported data type: vtktypeint32", "vtktypeint32"]
    
    def DisplayText(self, text):
        if not any(filtered in text for filtered in self.filtered_messages):
            sys.stderr.write(text)
    
    def DisplayErrorText(self, text):
        self.DisplayText(text)
    
    def DisplayWarningText(self, text):
        self.DisplayText(text)
    
    def DisplayGenericWarningText(self, text):
        self.DisplayText(text)

error_filter = VTKErrorFilter()
vtk.vtkOutputWindow.SetInstance(error_filter)

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def extract_attention_scores(model, surface_nodes_tensor, seg_matrix_tensor, vertices_tensor):
    """
    Extract attention scores from the cross-attention mechanism for each layer.
    Returns attention scores: list of (B, H, N, S) tensors, one per layer.
    """
    device = vertices_tensor.device
    attention_scores = []
    
    # Prepare inputs
    geo_coor = surface_nodes_tensor  # (B, M, 6)
    qcoor = vertices_tensor  # (B, N, 3)
    condition = None
    
    # Forward through preprocessing
    fx = model.preprocess(qcoor)
    fx = fx + model.placeholder[None, None, :]
    fx_surface = model.preprocess_surface(geo_coor)
    
    # Forward through each block and extract attention
    for i, block in enumerate(model.blocks):
        # Get the attention module
        attn_module = block.Attn
        
        # Apply layer norm first (as done in block.forward)
        fx_norm = block.ln_1(fx)
        fx_surface_norm = block.ln_1(fx_surface)
        
        # Prepare inputs for attention
        B, N, C = fx_norm.shape
        _, N_surface, _ = fx_surface_norm.shape
        
        # Extract point features
        fx_surface_proj = attn_module.in_project_surface(fx_surface_norm).reshape(
            B, N_surface, attn_module.heads, attn_module.dim_head
        ).permute(0, 2, 1, 3).contiguous()  # B H N_surface C
        
        fx_proj = attn_module.in_project_x(fx_norm).reshape(
            B, N, attn_module.heads, attn_module.dim_head
        ).permute(0, 2, 1, 3).contiguous()  # B H N C
        
        # Compute physics tokens
        tokens = torch.einsum("bsn,bhng->bhsg", seg_matrix_tensor, fx_surface_proj)  # B H S C
        
        # Compute queries and keys
        q_vol = attn_module.vol_to_q(fx_proj)  # B H N C
        k_slice_token = attn_module.surface_to_k(tokens)  # B H S C
        
        # Compute attention scores: q_vol @ k_slice_token^T / sqrt(d)
        # q_vol: (B, H, N, C), k_slice_token: (B, H, S, C)
        # scores: (B, H, N, S)
        scores = torch.matmul(q_vol, k_slice_token.transpose(-2, -1)) * attn_module.scale
        scores = attn_module.softmax(scores)  # Apply softmax
        
        attention_scores.append(scores.detach().cpu())  # Store attention scores
        
        # Continue forward pass through the block
        if i == len(model.blocks) - 1:
            fx = block(fx, fx_surface, seg_matrix_tensor)
        else:
            fx, fx_surface = block(fx, fx_surface, seg_matrix_tensor)
    
    return attention_scores


def compute_nodal_importance(attention_scores, seg_matrix, vertices, surface_nodes):
    """
    Compute nodal importance scores from attention scores.
    
    Args:
        attention_scores: List of attention score tensors, each (B, H, N, S)
        seg_matrix: (N_token, M) segmentation matrix mapping tokens to surface nodes
        vertices: (N, 3) volume node coordinates
        surface_nodes: (M, 6) surface node features (first 3 are coordinates)
    
    Returns:
        nodal_importance: (N,) array of importance scores
    """
    num_volume_nodes = len(vertices)
    num_tokens = seg_matrix.shape[0]
    
    # Map surface nodes to volume nodes: for each volume node, find closest surface node
    surface_coords = surface_nodes[:, :3]  # Extract coordinates from surface features
    distances = cdist(vertices, surface_coords)  # (N, M)
    closest_surface_idx = np.argmin(distances, axis=1)  # (N,) - index of closest surface node
    
    # For each token, determine which volume nodes belong to it
    # A volume node belongs to a token if its closest surface node belongs to that token
    # seg_matrix: (N_token, M) - each row is a token, each column is a surface node
    # Find which surface nodes belong to each token (non-zero entries in seg_matrix)
    token_to_volume_nodes = {}  # token_idx -> list of volume node indices
    
    for token_idx in range(num_tokens):
        # Get surface nodes that belong to this token (non-zero entries in seg_matrix[token_idx])
        token_surface_mask = seg_matrix[token_idx, :] > 0  # (M,) boolean mask
        token_surface_indices = np.where(token_surface_mask)[0]  # Indices of surface nodes for this token
        
        # Find volume nodes whose closest surface node is one of these token surface nodes
        volume_node_indices = np.where(np.isin(closest_surface_idx, token_surface_indices))[0]
        token_to_volume_nodes[token_idx] = volume_node_indices
    
    # Initialize importance to zero
    nodal_importance = np.zeros(num_volume_nodes)
    
    # For each layer's attention scores
    for layer_idx, attn_scores in enumerate(attention_scores):
        # attn_scores: (B, H, N, S) where N is volume nodes, S is number of tokens
        # Average over batch and heads: (N, S)
        attn_avg = attn_scores.squeeze(0).mean(dim=0).numpy()  # (N, S)
        
        # For each token
        for token_idx in range(num_tokens):
            # Get volume nodes that belong to this token
            volume_nodes_for_token = token_to_volume_nodes[token_idx]
            
            if len(volume_nodes_for_token) > 0:
                # For each volume node belonging to this token, add its attention score to this token
                for vol_node_idx in volume_nodes_for_token:
                    # Attention score from this volume node to this token
                    attention_score = attn_avg[vol_node_idx, token_idx]
                    nodal_importance[vol_node_idx] += attention_score
    
    return nodal_importance


def compute_importance_and_save_vtk(sample_id):
    """
    Compute nodal importance using attention scores from transolver_seg model.
    
    Steps:
    1. Find location of largest stress value
    2. Replace all volume coordinates with this coordinate
    3. Make prediction and extract attention scores
    4. Compute nodal importance from attention scores
    5. Save to VTK
    
    Args:
        sample_id: Sample ID string (e.g., '101_428')
    """
    model_name = 'transolver_seg'  # Must use transolver_seg
    
    # Set up paths
    data_root = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/JEB"
    processed_data_path = os.path.join(data_root, "Processed_seg2/")
    volume_mesh_path = os.path.join(data_root, "VolumeMesh")
    results_path = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/JEB/results/"
    model_path = f"trained_models_JEB/best_{model_name}_model.pth"
    
    os.makedirs(results_path, exist_ok=True)
    
    # 1) Load processed data to get exact values and seg_matrix
    processed_file = os.path.join(processed_data_path, f"{sample_id}.pkl")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data file not found: {processed_file}")
    
    print(f"Loading processed data from {processed_file}")
    with open(processed_file, 'rb') as f:
        sim_data = pickle.load(f)
    
    vertices = sim_data['volume_nodes']  # (N, 3)
    exact_stress = sim_data['volume_features']  # (N,) or (N, 1)
    surface_nodes = sim_data['surface_features']  # (M, 6)
    seg_matrix = sim_data['seg_matrix']  # (N_token, M) or csr_matrix
    
    # Convert seg_matrix if sparse
    if isinstance(seg_matrix, csr_matrix):
        seg_matrix = seg_matrix.toarray()
    
    # Ensure exact_stress is 1D
    if exact_stress.ndim > 1:
        exact_stress = exact_stress.squeeze()
    
    print(f"Loaded data: {len(vertices)} volume nodes, {len(surface_nodes)} surface nodes")
    print(f"Exact stress length: {len(exact_stress)} (should match volume nodes: {len(vertices)})")
    
    # Verify that exact_stress length matches vertices (accounting for padding)
    if len(exact_stress) != len(vertices):
        raise ValueError(
            f"Mismatch: exact_stress length ({len(exact_stress)}) != vertices length ({len(vertices)}). "
            f"This suggests the processed data may be corrupted or inconsistent."
        )
    
    # 2) Find location of largest stress value
    stress_mean, stress_std = _load_normalization_stats()
    exact_stress_denorm = exact_stress * stress_std + stress_mean
    
    max_stress_idx = np.argmax(exact_stress_denorm)
    max_stress_coord = vertices[max_stress_idx]  # (3,) - coordinate of max stress
    print(f"Maximum stress location: index {max_stress_idx}, coordinate {max_stress_coord}")
    print(f"Maximum stress value: {exact_stress_denorm[max_stress_idx]:.6f}")
    
    # 3) Replace all volume coordinates with the max stress coordinate
    num_volume_nodes = len(vertices)
    vertices_modified = np.tile(max_stress_coord, (1, 1))  # (1, 3) - all same coordinate
    print(f"Replaced all {num_volume_nodes} volume coordinates with one max stress coordinate")
    
    # 4) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model (must be transolver_seg)
    model = Transolver_seg_Model(
        space_dim=3,
        out_dim=1,
        n_layers=8,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=2,
    )
    
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.float().to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # 5) Prepare input tensors with modified coordinates
    vertices_modified_tensor = torch.from_numpy(vertices_modified).float().unsqueeze(0).to(device)  # (1, N, 3)
    surface_nodes_tensor = torch.from_numpy(surface_nodes).float().unsqueeze(0).to(device)  # (1, M, 6)
    seg_matrix_tensor = torch.from_numpy(seg_matrix).float().unsqueeze(0).to(device)  # (1, N_token, M)
    
    # 6) Make prediction and extract attention scores
    print("Making prediction with modified coordinates and extracting attention scores...")
    with torch.no_grad():
        attention_scores = extract_attention_scores(
            model, surface_nodes_tensor, seg_matrix_tensor, vertices_modified_tensor
        )
        # Also get the prediction
        prediction = model((surface_nodes_tensor, seg_matrix_tensor, vertices_modified_tensor, None))

    # compute the nodal importance from the attention scores
    def compute_nodal_importance(attention_scores, seg_matrix):
        noda_score = np.zeros(seg_matrix.shape[1])
        attention_scores = torch.mean(attention_scores, 1).squeeze(0).squeeze(0).numpy()
        for i in range(attention_scores.shape[0]):
            noda_score += np.where(seg_matrix[i,:] > 0, attention_scores[i], 0)
        return noda_score
    
    nodal_scores = []
    for i in range(len(attention_scores)):
        nodal_scores.append(compute_nodal_importance(attention_scores[i], seg_matrix))
    
    print(f"Computed nodal scores for {len(nodal_scores)} layers")
    for i, score in enumerate(nodal_scores):
        print(f"  Layer {i}: shape {score.shape}, range [{score.min():.6f}, {score.max():.6f}]")
    
    # 8) Read surface VTK file
    surface_vtk_path = os.path.join(data_root, "Processed_vtks2/")
    surface_vtk_file = os.path.join(surface_vtk_path, f"{sample_id}.vtk")
    
    if not os.path.exists(surface_vtk_file):
        raise FileNotFoundError(f"Surface VTK file not found: {surface_vtk_file}")
    
    print(f"Reading surface VTK file from {surface_vtk_file}")
    with suppress_stderr():
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(surface_vtk_file)
        reader.Update()
        surface_vtk_data = reader.GetOutput()
    
    # Surface VTK should be PolyData
    if not isinstance(surface_vtk_data, vtk.vtkPolyData):
        raise ValueError(f"Expected PolyData for surface VTK, got {type(surface_vtk_data)}")
    
    num_surface_points = surface_vtk_data.GetNumberOfPoints()
    print(f"Surface mesh has {num_surface_points} points")
    
    # Verify that number of points matches surface nodes
    if num_surface_points != len(surface_nodes):
        raise ValueError(
            f"Shape mismatch: Number of points in surface VTK ({num_surface_points}) does not match "
            f"number of surface nodes in processed data ({len(surface_nodes)}). "
            f"Please ensure you're using the correct surface VTK file."
        )
    
    # Verify all nodal scores have the correct length
    expected_surface_len = len(surface_nodes)
    for i, score in enumerate(nodal_scores):
        if len(score) != expected_surface_len:
            raise ValueError(
                f"Array length mismatch: Layer {i} nodal score should have length {expected_surface_len}, "
                f"but got {len(score)}"
            )
    
    # 9) Assign nodal scores from each layer to surface VTK
    point_data = surface_vtk_data.GetPointData()
    
    # Add nodal score for each layer
    for layer_idx, layer_score in enumerate(nodal_scores):
        score_vtk = numpy_support.numpy_to_vtk(layer_score, deep=True)
        score_vtk.SetName(f"NodalImportance_Layer{layer_idx}")
        point_data.AddArray(score_vtk)
        print(f"Added nodal feature: NodalImportance_Layer{layer_idx}")
    
    # Also add aggregated score (sum across all layers)
    aggregated_score = np.sum(nodal_scores, axis=0)  # (M,) - sum across layers
    aggregated_vtk = numpy_support.numpy_to_vtk(aggregated_score, deep=True)
    aggregated_vtk.SetName("NodalImportance_AllLayers")
    point_data.AddArray(aggregated_vtk)
    
    # Set aggregated score as active scalars for visualization
    point_data.SetActiveScalars("NodalImportance_AllLayers")
    
    print(f"Added aggregated nodal feature: NodalImportance_AllLayers")
    print(f"  Aggregated score range: [{aggregated_score.min():.6f}, {aggregated_score.max():.6f}]")
    print(f"  Mean aggregated score: {aggregated_score.mean():.6f}")
    
    # 10) Compute closeness to highest stress location
    print(f"\nComputing closeness to highest stress location...")
    surface_coords = surface_nodes[:, :3]  # Extract coordinates from surface features (first 3 are coords)
    
    # Compute distances from max stress coordinate to all surface nodes
    distances_to_max_stress = np.linalg.norm(surface_coords - max_stress_coord, axis=1)  # (M,)
    
    # Find the 100 closest surface nodes
    num_closest = min(100, len(surface_coords))
    closest_indices = np.argsort(distances_to_max_stress)[:num_closest]
    
    # Create binary array: 1 for closest 100 nodes, 0 for others
    closeness_binary = np.zeros(len(surface_coords), dtype=np.int32)
    closeness_binary[closest_indices] = 1
    
    print(f"  Found {num_closest} closest surface nodes to max stress location")
    print(f"  Distance range to max stress: [{distances_to_max_stress.min():.6f}, {distances_to_max_stress.max():.6f}]")
    print(f"  Distance to {num_closest}th closest node: {distances_to_max_stress[closest_indices[-1]]:.6f}")
    
    # Add closeness feature to VTK
    closeness_vtk = numpy_support.numpy_to_vtk(closeness_binary, deep=True)
    closeness_vtk.SetName("ClosenessToMaxStress")
    point_data.AddArray(closeness_vtk)
    
    print(f"Added nodal feature: ClosenessToMaxStress (1 for {num_closest} closest nodes, 0 for others)")
    
    # 10) Save surface VTK file
    output_file = os.path.join(results_path, f"{sample_id}_{model_name}_importance_surface.vtk")
    print(f"Saving surface VTK file to {output_file}")
    
    with suppress_stderr():
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(surface_vtk_data)
        writer.Write()
    
    print(f"✅ Successfully saved surface VTK file: {output_file}")
    print(f"   - Number of layers: {len(nodal_scores)}")
    print(f"   - Aggregated importance range: [{aggregated_score.min():.6f}, {aggregated_score.max():.6f}]")
    print(f"\n📍 Location of highest stress:")
    print(f"   - Node index: {max_stress_idx}")
    print(f"   - Coordinate (X, Y, Z): ({max_stress_coord[0]:.6f}, {max_stress_coord[1]:.6f}, {max_stress_coord[2]:.6f})")
    print(f"   - Stress value: {exact_stress_denorm[max_stress_idx]:.6f}")


if __name__ == "__main__":
    # Get inputs from command line
    Test_ID = '101_428'
    
    # Allow override from command line
    if len(sys.argv) > 1:
        Test_ID = sys.argv[1]
    
    print(f"Processing sample ID: {Test_ID} with transolver_seg model for importance analysis")
    compute_importance_and_save_vtk(Test_ID)
