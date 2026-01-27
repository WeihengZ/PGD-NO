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


def compute_error_and_save_vtk(sample_id, model_name):
    """
    Compute pointwise absolute error between prediction and exact values,
    assign cluster labels from seg matrix, and save VTK with nodal features.
    
    Args:
        sample_id: Sample ID string (e.g., '101_428')
        model_name: Model name ('transolver' or 'transolver_seg')
    """
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
    
    # 2) Load model and make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if model_name == "transolver":
        model = Transolver_Model(
            space_dim=3,
            out_dim=1,
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
            slice_num=64,
        )
    elif model_name == "transolver_seg":
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
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.float().to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Prepare input tensors
    vertices_tensor = torch.from_numpy(vertices).float().unsqueeze(0).to(device)  # (1, N, 3)
    surface_nodes_tensor = torch.from_numpy(surface_nodes).float().unsqueeze(0).to(device)  # (1, M, 6)
    seg_matrix_tensor = torch.from_numpy(seg_matrix).float().unsqueeze(0).to(device)  # (1, N_token, M)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        if model_name == "transolver":
            prediction = model(vertices_tensor)  # (1, N, 1)
        elif model_name == "transolver_seg":
            prediction = model((surface_nodes_tensor, seg_matrix_tensor, vertices_tensor, None))  # (1, N, 1)
    
    prediction = prediction.squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
    
    # Load normalization stats and denormalize
    stress_mean, stress_std = _load_normalization_stats()
    prediction_denorm = prediction * stress_std + stress_mean
    exact_stress_denorm = exact_stress * stress_std + stress_mean
    
    # 3) Compute pointwise absolute error
    error = np.abs(prediction_denorm - exact_stress_denorm)
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
    
    # 4) Assign cluster labels from seg_matrix
    # seg_matrix is (N_token, M) where M is number of surface nodes
    # We need to map surface node labels to volume nodes
    
    # Get cluster labels for surface nodes
    seg_matrix_dense = seg_matrix  # Already converted to array
    surface_node_labels = np.argmax(seg_matrix_dense, axis=0)  # (M,) - cluster ID for each surface node
    
    # Map surface node labels to volume nodes
    # Strategy: For each volume node, find the closest surface node and assign its label
    surface_coords = surface_nodes[:, :3]  # Extract coordinates from surface features (first 3 are coords)
    
    # Compute distances from volume nodes to surface nodes
    distances = cdist(vertices, surface_coords)  # (N, M)
    closest_surface_idx = np.argmin(distances, axis=1)  # (N,) - index of closest surface node for each volume node
    volume_cluster_labels = surface_node_labels[closest_surface_idx]  # (N,) - cluster label for each volume node
    
    # Now assign new cluster IDs: sort by cluster size, assign maximally different IDs to large clusters
    num_clusters = seg_matrix_dense.shape[0]
    cluster_sizes = np.bincount(volume_cluster_labels, minlength=num_clusters)  # Size of each cluster
    
    # Sort clusters by size (largest first)
    sorted_cluster_indices = np.argsort(cluster_sizes)[::-1]  # Indices sorted by size (largest to smallest)
    
    # Determine how many clusters to assign maximally different IDs to
    # Use top clusters that together represent a significant portion, or top N clusters
    # For now, let's use top clusters that represent at least 50% of nodes, or top 24, whichever is larger
    cumulative_sizes = np.cumsum(cluster_sizes[sorted_cluster_indices])
    total_nodes = len(volume_cluster_labels)
    # Find clusters that together represent at least 50% of nodes
    top_clusters_mask = cumulative_sizes <= (total_nodes * 0.5)
    num_top_from_50pct = np.sum(top_clusters_mask) if np.any(top_clusters_mask) else 0
    num_top_clusters = max(num_top_from_50pct, min(24, num_clusters))
    num_top_clusters = min(num_top_clusters, num_clusters)  # Ensure we don't exceed total clusters
    
    # For top (large) clusters: assign maximally different IDs
    # Strategy: Space them out evenly in the ID space [1, num_clusters]
    if num_top_clusters > 0:
        # Create maximally spaced IDs in the range [1, num_clusters]
        # For example, if num_clusters=100 and num_top_clusters=5, we want: 1, 25, 50, 75, 100
        if num_top_clusters == 1:
            top_ids = [1]
        elif num_top_clusters == 2:
            top_ids = [1, num_clusters]
        else:
            # Space IDs evenly across the ID range [1, num_clusters]
            step = (num_clusters - 1) / (num_top_clusters - 1)
            top_ids = [int(round(1 + i * step)) for i in range(num_top_clusters)]
            # Ensure IDs are within valid range [1, num_clusters] and unique
            top_ids = [max(1, min(id_val, num_clusters)) for id_val in top_ids]
            top_ids = sorted(list(set(top_ids)))  # Remove duplicates and sort
            # If we lost some IDs due to rounding/duplicates, fill gaps
            while len(top_ids) < num_top_clusters:
                # Find a gap and fill it
                all_ids_set = set(top_ids)
                for candidate in range(1, num_clusters + 1):
                    if candidate not in all_ids_set:
                        top_ids.append(candidate)
                        break
                top_ids = sorted(list(set(top_ids)))  # Re-sort after adding
            top_ids = sorted(top_ids[:num_top_clusters])  # Take exactly num_top_clusters and sort
    else:
        top_ids = []
    
    # For remaining (smaller) clusters: uniformly sample IDs from [1, num_clusters]
    num_remaining = num_clusters - num_top_clusters
    if num_remaining > 0:
        # Get all IDs not used by top clusters
        all_ids = list(range(1, num_clusters + 1))
        remaining_ids = [id for id in all_ids if id not in top_ids]
        # Uniformly sample from remaining IDs
        remaining_cluster_ids = np.random.choice(remaining_ids, size=num_remaining, replace=False).tolist()
    else:
        remaining_cluster_ids = []
    
    # Create mapping: original cluster ID -> new sampled ID
    label_map = np.zeros(num_clusters, dtype=np.int32)
    
    # Assign maximally different IDs to top clusters
    for idx, original_cluster_id in enumerate(sorted_cluster_indices[:num_top_clusters]):
        label_map[original_cluster_id] = top_ids[idx]
    
    # Assign uniformly sampled IDs to remaining clusters
    for idx, original_cluster_id in enumerate(sorted_cluster_indices[num_top_clusters:]):
        label_map[original_cluster_id] = remaining_cluster_ids[idx]
    
    # Apply the mapping to volume cluster labels
    cluster_labels = label_map[volume_cluster_labels]
    
    print(f"Number of unique clusters: {num_clusters}")
    print(f"Top {num_top_clusters} clusters (largest) assigned maximally different IDs: {sorted(top_ids)}")
    print(f"Remaining {num_remaining} clusters assigned uniformly sampled IDs")
    print(f"Cluster label range: [{cluster_labels.min()}, {cluster_labels.max()}]")
    
    # 5) Read raw VTK file (keep original structure - don't convert)
    volume_file = os.path.join(volume_mesh_path, f"{sample_id}.vtk")
    if not os.path.exists(volume_file):
        raise FileNotFoundError(f"Volume mesh file not found: {volume_file}")
    
    print(f"Reading VTK file from {volume_file}")
    with suppress_stderr():
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(volume_file)
        reader.Update()
        data_object = reader.GetOutput()
    
    # Keep the original VTK structure (UnstructuredGrid or PolyData)
    # We're saving VOLUME predictions directly to the volume VTK
    if isinstance(data_object, vtk.vtkUnstructuredGrid):
        vtk_data = data_object
        num_points = vtk_data.GetNumberOfPoints()
        print(f"Volume mesh (UnstructuredGrid) has {num_points} points")
    elif isinstance(data_object, vtk.vtkPolyData):
        vtk_data = data_object
        num_points = vtk_data.GetNumberOfPoints()
        print(f"Volume mesh (PolyData) has {num_points} points")
    else:
        raise ValueError(f"Unsupported VTK data type: {type(data_object)}")
    
    # Verify that number of points matches exactly - raise error if not
    # Note: The processed data's volume_features is padded with 5 zeros to match VTK
    # So num_points should equal len(vertices) if using the same VTK file
    if num_points != len(vertices):
        raise ValueError(
            f"Shape mismatch: Number of points in VTK ({num_points}) does not match "
            f"number of vertices in processed data ({len(vertices)}). "
            f"This indicates a mismatch between the VTK file used for processing and the one being read. "
            f"Please ensure you're using the same VTK file that was used during data processing."
        )
    
    # Verify all arrays have the correct length
    expected_len = len(vertices)
    if (len(prediction_denorm) != expected_len or 
        len(exact_stress_denorm) != expected_len or 
        len(error) != expected_len or 
        len(cluster_labels) != expected_len):
        raise ValueError(
            f"Array length mismatch: All arrays should have length {expected_len}, but got: "
            f"prediction={len(prediction_denorm)}, exact={len(exact_stress_denorm)}, "
            f"error={len(error)}, cluster_labels={len(cluster_labels)}"
        )
    
    # 6) Assign nodal features to VTK data
    prediction_vtk = numpy_support.numpy_to_vtk(prediction_denorm, deep=True)
    prediction_vtk.SetName("Prediction")
    
    exact_vtk = numpy_support.numpy_to_vtk(exact_stress_denorm, deep=True)
    exact_vtk.SetName("Exact")
    
    error_vtk = numpy_support.numpy_to_vtk(error, deep=True)
    error_vtk.SetName("Error")
    
    cluster_labels_vtk = numpy_support.numpy_to_vtk(cluster_labels.astype(np.int32), deep=True)
    cluster_labels_vtk.SetName("ClusterLabel")
    
    # Add arrays to point data
    point_data = vtk_data.GetPointData()
    point_data.AddArray(prediction_vtk)
    point_data.AddArray(exact_vtk)
    point_data.AddArray(error_vtk)
    point_data.AddArray(cluster_labels_vtk)
    
    # Set Prediction as active scalars for visualization
    point_data.SetActiveScalars("Prediction")
    
    print("Added nodal features: Prediction, Exact, Error, ClusterLabel")
    
    # 7) Save VTK file (using appropriate writer for the data type)
    output_file = os.path.join(results_path, f"{sample_id}_{model_name}.vtk")
    print(f"Saving VTK file to {output_file}")
    
    with suppress_stderr():
        if isinstance(vtk_data, vtk.vtkUnstructuredGrid):
            writer = vtk.vtkUnstructuredGridWriter()
        elif isinstance(vtk_data, vtk.vtkPolyData):
            writer = vtk.vtkPolyDataWriter()
        else:
            raise ValueError(f"Cannot determine writer for VTK data type: {type(vtk_data)}")
        
        writer.SetFileName(output_file)
        writer.SetInputData(vtk_data)
        writer.Write()
    
    print(f"✅ Successfully saved VTK file: {output_file}")
    print(f"   - Prediction range: [{prediction_denorm.min():.6f}, {prediction_denorm.max():.6f}]")
    print(f"   - Exact range: [{exact_stress_denorm.min():.6f}, {exact_stress_denorm.max():.6f}]")
    print(f"   - Error range: [{error.min():.6f}, {error.max():.6f}]")
    print(f"   - Cluster labels: {len(np.unique(cluster_labels))} unique clusters")


if __name__ == "__main__":
    # Get inputs from test.py file or command line
    Test_ID = '101_428'
    model_name = 'transolver_seg'
    
    # Allow override from command line
    if len(sys.argv) > 1:
        Test_ID = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    
    print(f"Processing sample ID: {Test_ID}, Model: {model_name}")
    compute_error_and_save_vtk(Test_ID, model_name)
