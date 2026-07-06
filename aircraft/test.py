import os
import sys
import glob
import h5py
import numpy as np
import torch
import vtk
from vtk.util import numpy_support
from contextlib import contextmanager
from scipy.sparse import csr_matrix

# Import model classes
from models.Transolver_plus import Model as Transolver_plus
from models.Transolver_seg import Model as Transolver_seg
from models.Transolver_seg_v2 import Model as Transolver_seg_v2
from models.Transolver_seg_v3 import Model as Transolver_seg_v3
from models.Transolver_seg_v4 import Model as Transolver_seg_v4
from models.Transolver_seg_v5 import Model as Transolver_seg_v5

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


def compute_node_labels_from_seg_matrix(seg_matrix, num_nodes):
    """
    Compute node labels from seg_matrix for geometry tokens.
    Uses argmax to assign nodes to clusters, then assigns maximally different IDs
    to large clusters for better visualization.
    
    Args:
        seg_matrix: (N_token, N_nodes) or (B, N_token, N_nodes) numpy array or csr_matrix
        num_nodes: Number of nodes
        
    Returns:
        numpy.ndarray: Node labels of shape (num_nodes,)
    """
    # Handle batch dimension
    if seg_matrix.ndim == 3:
        seg_matrix = seg_matrix[0]  # Take first batch
    
    # Convert sparse to dense if needed
    if isinstance(seg_matrix, csr_matrix):
        seg_matrix = seg_matrix.toarray()
    
    # Ensure seg_matrix is (N_token, N_nodes)
    if seg_matrix.shape[0] == num_nodes:
        seg_matrix = seg_matrix.T  # Transpose if (N, N_token) -> (N_token, N)
    
    # seg_matrix is (N_token, N_nodes)
    # Assign each node to the token with maximum value (argmax along token axis)
    node_cluster_ids = np.argmax(seg_matrix, axis=0)  # (N_nodes,) - cluster ID for each node
    
    num_clusters = seg_matrix.shape[0]
    
    # Compute cluster sizes
    cluster_sizes = np.bincount(node_cluster_ids, minlength=num_clusters)  # Size of each cluster
    
    # Sort clusters by size (largest first)
    sorted_cluster_indices = np.argsort(cluster_sizes)[::-1]  # Indices sorted by size (largest to smallest)
    
    # Determine how many clusters to assign maximally different IDs to
    # Use top clusters that together represent at least 50% of nodes, or top 24, whichever is larger
    cumulative_sizes = np.cumsum(cluster_sizes[sorted_cluster_indices])
    total_nodes = len(node_cluster_ids)
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
    
    # Apply the mapping to node cluster labels
    node_labels = label_map[node_cluster_ids]
    
    # Print cluster assignment information
    print(f"   Number of unique clusters: {num_clusters}")
    print(f"   Top {num_top_clusters} clusters (largest) assigned maximally different IDs: {sorted(top_ids)}")
    print(f"   Remaining {num_remaining} clusters assigned uniformly sampled IDs")
    print(f"   Cluster label range: [{node_labels.min()}, {node_labels.max()}]")
    
    return node_labels


def create_vtk_from_data(pos, faces=None):
    """
    Create VTK polydata from position data.
    
    Args:
        pos: (N, 3) numpy array of positions
        faces: (M, 3) numpy array of face indices, optional
        
    Returns:
        vtk.vtkPolyData: VTK polydata object
    """
    coords = np.array(pos, dtype=np.float64)
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(coords))
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Add faces if provided
    if faces is not None:
        vtk_cells = vtk.vtkCellArray()
        for face in faces:
            face_array = np.array(face, dtype=np.int64)
            vtk_cells.InsertNextCell(len(face_array))
            for vertex_id in face_array:
                vtk_cells.InsertCellPoint(int(vertex_id))
        polydata.SetPolys(vtk_cells)
    
    return polydata


def find_h5_file(data_dir, idx):
    """
    Find H5 file for given sample index by searching for files starting with '{idx}_'.
    
    Args:
        data_dir: Directory containing H5 files
        idx: Sample index (integer)
        
    Returns:
        tuple: (h5_file_path, Ma, alpha, beta) from file attributes
    """
    pattern = os.path.join(data_dir, f'{int(idx)}_*.h5')
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No H5 file found for sample ID {idx} in {data_dir}")
    
    if len(matching_files) > 1:
        print(f"Warning: Multiple H5 files found for sample ID {idx}. Using first match: {matching_files[0]}")
    
    h5_file = matching_files[0]
    
    # Read Ma, alpha, beta from file attributes
    with h5py.File(h5_file, 'r') as f:
        Ma = float(f.attrs['Ma'])
        alpha = float(f.attrs['alpha'])
        beta = float(f.attrs['beta'])
    
    return h5_file, Ma, alpha, beta


def compute_error_and_save_vtk(idx, model_name, model_path=None):
    """
    Compute pointwise absolute error between prediction and exact values,
    assign cluster labels from seg matrix (for seg models), and save VTK with nodal features.
    
    Args:
        idx: Sample index (integer)
        model_name: Model name ('Transolver_plus', 'Transolver_seg', 'Transolver_seg_v2', etc.)
        model_path: Path to model file (optional, defaults to './trained_models/best_{model_name}_model.pth')
    """
    # Set up paths
    data_dir = '/taiga/illinois/eng/cee/meidani/Vincent/aircraft_industry/processed_data3/'
    results_path = '/taiga/illinois/eng/cee/meidani/Vincent/aircraft_industry/results/'
    
    # Auto-determine model path from model name (matching main_airplane.py)
    if model_path is None:
        model_path = f'./trained_models/best_{model_name}_model.pth'
    
    os.makedirs(results_path, exist_ok=True)
    
    # 1) Find and load data from H5 file
    h5_file, Ma, alpha, beta = find_h5_file(data_dir, idx)
    
    print(f"Loading data from {h5_file}")
    print(f"Sample parameters: Ma={Ma}, alpha={alpha}, beta={beta}")
    
    with h5py.File(h5_file, 'r') as f:
        pos = f['pos'][:]  # (N, 3)
        normals = f['normals'][:]  # (N, 3)
        values = f['values'][:]  # (N, 6) - exact values
        seg_matrix = f['seg_matrix'][:]  # (N_token, N) or (N, N_token)
        # Check if faces/connectivity data exists
        faces = None
        if 'faces' in f:
            faces = f['faces'][:]  # (M, 3) face connectivity
            print(f"Found faces in H5 file: {faces.shape}")
        elif 'connectivity' in f:
            faces = f['connectivity'][:]  # Alternative name
            print(f"Found connectivity in H5 file: {faces.shape}")
        elif 'cells' in f:
            faces = f['cells'][:]  # Alternative name
            print(f"Found cells in H5 file: {faces.shape}")
    
    num_nodes = len(pos)
    print(f"Loaded data: {num_nodes} nodes")
    print(f"Exact values shape: {values.shape}")
    
    # Ensure seg_matrix is in correct format (N_token, N)
    if seg_matrix.shape[0] == num_nodes:
        seg_matrix = seg_matrix.T  # Transpose if (N, N_token) -> (N_token, N)
    
    # 2) Normalization constants (from main_airplane.py)
    pos_mean = torch.tensor([2.80879162e+03, 1.00957077e+02, 6.76594237e-03]).view(1, 1, 3)
    pos_std = torch.tensor([1436.65326859, 178.37956359, 615.16521715]).view(1, 1, 3)
    norm_mean = torch.tensor([-7.03865828e-02, 1.50757955e-01, -6.07368549e-06]).view(1, 1, 3)
    norm_std = torch.tensor([0.19895465, 0.87515866, 0.40665163]).view(1, 1, 3)
    out_mean = torch.tensor([0.04602036, 1.3157164, 5.66693757, 0.25599, 0.06231503, 1.64027649]).view(1, 1, 6)
    out_std = torch.tensor([0.09458788, 0.76978003, 0.41717544, 0.47068753, 0.6710297, 1.8059161]).view(1, 1, 6)
    
    # 3) Load model and make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if model_name == 'Transolver_plus':
        model = Transolver_plus(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    elif model_name == 'Transolver_seg':
        model = Transolver_seg(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    elif model_name == 'Transolver_seg_v2':
        model = Transolver_seg_v2(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    elif model_name == 'Transolver_seg_v3':
        model = Transolver_seg_v3(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    elif model_name == 'Transolver_seg_v4':
        model = Transolver_seg_v4(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    elif model_name == 'Transolver_seg_v5':
        model = Transolver_seg_v5(n_hidden=256, n_layers=4, space_dim=7,
                    fun_dim=0,
                    n_head=8,
                    mlp_ratio=2, out_dim=6,
                    slice_num=32,
                    unified_pos=0,
                    dropout=0.1).to(device)
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Handle DDP wrapper if present
    model_state = torch.load(model_path, map_location=device)
    if isinstance(model_state, torch.nn.Module):
        # If the saved model is wrapped in DDP, extract state_dict
        if hasattr(model_state, 'module'):
            model.load_state_dict(model_state.module.state_dict())
        else:
            model.load_state_dict(model_state.state_dict())
    else:
        # If it's a state_dict
        model.load_state_dict(model_state)
    
    model = model.float().to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Prepare input tensors
    pos_tensor = torch.from_numpy(pos).float().unsqueeze(0).to(device)  # (1, N, 3)
    normals_tensor = torch.from_numpy(normals).float().unsqueeze(0).to(device)  # (1, N, 3)
    
    # Normalize positions
    pos_normalized = (pos_tensor - pos_mean.to(device)) / pos_std.to(device)
    
    # Create input x: [pos, sdf, normals] where sdf is zeros
    N = pos_tensor.shape[1]
    sdf = torch.zeros((1, N, 1), dtype=torch.float32).to(device)
    x = torch.cat([pos_normalized, sdf, normals_tensor], dim=2)  # (1, N, 7)
    
    # Create condition tensor
    condition = torch.tensor([Ma, alpha, beta]).view(1, 3).to(device).float()
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        if model_name == 'Transolver_plus':
            prediction = model((x, pos_normalized, condition))  # (1, N, 6)
        elif model_name[:14] == 'Transolver_seg':
            seg_matrix_tensor = torch.from_numpy(seg_matrix).float().unsqueeze(0).to(device)  # (1, N_token, N)
            # Create inverse seg matrix
            inverse_seg_matrix = seg_matrix_tensor.permute(0, 2, 1)  # (1, N, N_token)
            inverse_seg_matrix = torch.where(inverse_seg_matrix > 0, 1.0, 0.0).float()
            prediction = model((x, pos_normalized, condition, seg_matrix_tensor, inverse_seg_matrix))  # (1, N, 6)
        else:
            raise ValueError(f"Model name {model_name} not supported")
    
    # Denormalize prediction
    prediction_denorm = prediction * out_std.to(device) + out_mean.to(device)
    prediction_denorm = prediction_denorm.squeeze(0).cpu().numpy()  # (N, 6)
    
    # Exact values are already in original scale
    exact_values = values  # (N, 6)
    
    # 4) Compute pointwise absolute error for each output dimension
    error = np.abs(prediction_denorm - exact_values)  # (N, 6)
    
    # For visualization, we'll use the last dimension (index -1) as the main scalar
    # You can modify this to use a different dimension if needed
    prediction_scalar = prediction_denorm[:, -1]  # Last dimension
    exact_scalar = exact_values[:, -1]
    error_scalar = error[:, -1]
    
    print(f"Mean absolute error (last dim): {error_scalar.mean():.6f}")
    print(f"Max absolute error (last dim): {error_scalar.max():.6f}")
    
    # 5) Compute node labels for seg models
    node_labels = None
    if model_name[:14] == 'Transolver_seg' and seg_matrix is not None:
        print("Computing node labels from seg_matrix...")
        node_labels = compute_node_labels_from_seg_matrix(seg_matrix, num_nodes)
        print(f"✅ Computed node labels: {len(np.unique(node_labels))} unique labels")
    
    # 6) Create VTK polydata from positions
    print("Creating VTK polydata...")
    if faces is not None:
        # Use face connectivity if available
        vtk_data = create_vtk_from_data(pos, faces)
        print(f"Created VTK polydata with {len(faces)} faces")
    else:
        # Create point cloud with vertex cells so ParaView can render them
        vtk_data = create_vtk_from_data(pos, faces=None)
        # Add vertex cells for each point so ParaView can render them
        vtk_vertices = vtk.vtkCellArray()
        for i in range(num_nodes):
            vtk_vertices.InsertNextCell(1)
            vtk_vertices.InsertCellPoint(i)
        vtk_data.SetVerts(vtk_vertices)
        print(f"Created VTK polydata with {num_nodes} points (as vertices)")
    
    num_points = vtk_data.GetNumberOfPoints()
    print(f"VTK file has {num_points} points")
    
    # Verify that number of points matches
    if num_points != num_nodes:
        raise ValueError(
            f"Shape mismatch: Number of points in VTK ({num_points}) does not match "
            f"number of nodes in data ({num_nodes})."
        )
    
    # 7) Assign nodal features to VTK data
    prediction_vtk = numpy_support.numpy_to_vtk(prediction_scalar, deep=True)
    prediction_vtk.SetName("Prediction")
    
    exact_vtk = numpy_support.numpy_to_vtk(exact_scalar, deep=True)
    exact_vtk.SetName("Exact")
    
    error_vtk = numpy_support.numpy_to_vtk(error_scalar, deep=True)
    error_vtk.SetName("Error")
    
    # Add arrays to point data
    point_data = vtk_data.GetPointData()
    point_data.AddArray(prediction_vtk)
    point_data.AddArray(exact_vtk)
    point_data.AddArray(error_vtk)
    
    # Add node labels if available (for seg models)
    if node_labels is not None:
        node_labels_vtk = numpy_support.numpy_to_vtk(node_labels.astype(np.int32), deep=True)
        node_labels_vtk.SetName("node_labels")
        point_data.AddArray(node_labels_vtk)
        print("Added node_labels array for geometry tokens")
    
    # Set prediction as active scalars for visualization
    point_data.SetActiveScalars("Prediction")
    
    print(f"Added nodal features: Prediction, Exact, Error")
    if node_labels is not None:
        print(f"  - node_labels (geometry token labels)")
    
    # 8) Save VTK file
    output_file = os.path.join(results_path, f"{int(idx)}_{Ma}_{alpha}_{beta}_{model_name}.vtk")
    print(f"Saving VTK file to {output_file}")
    
    with suppress_stderr():
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(vtk_data)
        writer.SetFileTypeToBinary()  # Use binary format
        writer.Write()
    
    print(f"✅ Successfully saved VTK file: {output_file}")
    print(f"   - Prediction range: [{prediction_scalar.min():.6f}, {prediction_scalar.max():.6f}]")
    print(f"   - Exact range: [{exact_scalar.min():.6f}, {exact_scalar.max():.6f}]")
    print(f"   - Error range: [{error_scalar.min():.6f}, {error_scalar.max():.6f}]")
    if node_labels is not None:
        print(f"   - Node labels: {len(np.unique(node_labels))} unique labels")
    
    # ParaView rendering tips
    if faces is None:
        print(f"\n📌 ParaView Tips:")
        print(f"   - The file contains {num_nodes} points as vertices")
        print(f"   - To see the geometry: Apply 'Glyph' filter and set Glyph Type to 'Sphere'")
        print(f"   - Or: In Properties panel, increase 'Point Size' (if available)")
        print(f"   - Use 'Coloring' to visualize Prediction, Exact, or Error arrays")


if __name__ == "__main__":
    # Get inputs from command line or use defaults
    # Usage: python test.py <sample_id> [model_name] [model_path]
    idx = 23
    model_name = 'Transolver_seg_v2'
    model_path = None  # Will be auto-determined from model_name
    
    # Allow override from command line
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    
    print(f"Processing: Sample ID={idx}, Model: {model_name}")
    if model_path:
        print(f"Using custom model path: {model_path}")
    else:
        print(f"Using default model path: ./trained_models/best_{model_name}_model.pth")
    
    compute_error_and_save_vtk(idx, model_name, model_path)

