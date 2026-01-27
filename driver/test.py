import os
import sys
import pickle
import numpy as np
import torch
import vtk
from vtk.util import numpy_support
from contextlib import contextmanager
from scipy.sparse import csr_matrix

# Import model classes
from models.transolver_model import Model as Transolver_Model
from models.Transolver_seg import Model as Transolver_SEG_Model
from models.Transolver_seg_v2 import Model as Transolver_SEG_V2_Model

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
    
    # seg_matrix is (N_token, N_nodes)
    # Assign each node to the token with maximum value
    node_labels = np.zeros(num_nodes, dtype=np.int32)
    
    # Create random labels for tokens (1 to N_token)
    num_tokens = seg_matrix.shape[0]
    random_label = np.arange(1, num_tokens + 1)
    np.random.shuffle(random_label)
    
    # For each token, assign its label to nodes where it has non-zero values
    for i in range(num_tokens):
        # Find indices where this token has non-zero values
        non_zero_indices = np.nonzero(seg_matrix[i])[0]
        if len(non_zero_indices) > 0:
            node_labels[non_zero_indices] = random_label[i]
    
    return node_labels


def compute_error_and_save_vtk(sample_id, model_name, predicted_feature_name="pressure"):
    """
    Compute pointwise absolute error between prediction and exact values,
    assign cluster labels from seg matrix (for seg models), and save VTK with nodal features.
    
    Args:
        sample_id: Sample ID (integer, e.g., 46)
        model_name: Model name ('transolver', 'transolver_seg', or 'transolver_seg_v2')
        predicted_feature_name: Feature name to predict (default: 'pressure')
    """
    # Set up paths
    data_root = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/normalized/"
    original_vtk_path = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/selected/"
    results_path = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/results/"
    model_flag = f"{model_name}_{predicted_feature_name}"
    model_path = f"trained_models/best_model_{model_flag}.pth"
    
    os.makedirs(results_path, exist_ok=True)
    
    # 1) Load processed data
    processed_file = os.path.join(data_root, f"{sample_id}.pkl")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data file not found: {processed_file}")
    
    print(f"Loading processed data from {processed_file}")
    with open(processed_file, 'rb') as f:
        sample_data = pickle.load(f)
    
    features_6d = sample_data['features_6d']  # (N, 6) - coordinates + normals
    exact_target = sample_data[predicted_feature_name]  # (N,) or (N, 1)
    seg_matrix = sample_data.get('seg_matrix', None)  # (N_token, N) or csr_matrix
    
    # Ensure exact_target is 1D
    if exact_target.ndim > 1:
        exact_target = exact_target.squeeze()
    
    num_nodes = len(features_6d)
    print(f"Loaded data: {num_nodes} nodes")
    print(f"Exact target length: {len(exact_target)} (should match nodes: {num_nodes})")
    
    # Verify that exact_target length matches nodes
    if len(exact_target) != num_nodes:
        raise ValueError(
            f"Mismatch: exact_target length ({len(exact_target)}) != nodes length ({num_nodes}). "
            f"This suggests the processed data may be corrupted or inconsistent."
        )
    
    # 2) Load normalization scalars
    normalization_file = os.path.join(data_root, "normalization_scalars.pkl")
    if not os.path.exists(normalization_file):
        raise FileNotFoundError(f"Normalization scalars file not found: {normalization_file}")
    
    with open(normalization_file, 'rb') as f:
        normalization_scalars = pickle.load(f)
    
    # 3) Load model and make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    if model_name == "transolver":
        model = Transolver_Model(
            space_dim=6,
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
        model = Transolver_SEG_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
        )
    elif model_name == "transolver_seg_v2":
        model = Transolver_SEG_V2_Model(
            space_dim=6,
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
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.float().to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Prepare input tensors
    features_6d_tensor = torch.from_numpy(features_6d).float().unsqueeze(0).to(device)  # (1, N, 6)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        if model_name == "transolver":
            prediction = model(features_6d_tensor)  # (1, N, 1)
        elif model_name == "transolver_seg" or model_name == "transolver_seg_v2":
            if seg_matrix is None:
                raise ValueError(f"seg_matrix is required for {model_name} model but not found in data")
            # Convert seg_matrix to tensor
            if isinstance(seg_matrix, csr_matrix):
                seg_matrix = seg_matrix.toarray()
            seg_matrix_tensor = torch.from_numpy(seg_matrix).float().unsqueeze(0).to(device)  # (1, N_token, N)
            prediction = model((features_6d_tensor, seg_matrix_tensor))  # (1, N, 1)
        else:
            raise ValueError(f"Model name {model_name} not supported")
    
    prediction = prediction.squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
    
    # Denormalize predictions and targets
    from utils.metric import denormalize_pressure
    
    prediction_denorm = denormalize_pressure(torch.from_numpy(prediction).unsqueeze(-1), normalization_scalars).squeeze().numpy()
    exact_target_denorm = denormalize_pressure(torch.from_numpy(exact_target).unsqueeze(-1), normalization_scalars).squeeze().numpy()
    
    # 4) Compute pointwise absolute error
    error = np.abs(prediction_denorm - exact_target_denorm)
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
    
    # 5) Compute node labels for seg models
    node_labels = None
    if model_name in ["transolver_seg", "transolver_seg_v2"] and seg_matrix is not None:
        node_labels = compute_node_labels_from_seg_matrix(seg_matrix, num_nodes)
        print(f"Computed node labels: {len(np.unique(node_labels))} unique labels")
    
    # 6) Read original VTK file
    # Try to find the original VTK file
    # The pattern is F_D_WM_WW_{number}_F_D_WM_WW_{sample_id:04d}.vtk
    # where the first number can vary (1, 3, 5, 8, etc.)
    original_vtk_file = None
    sample_id_str = f"{sample_id:04d}"
    
    # Try common numbers first, then search all files if needed
    possible_numbers = [1, 3, 5, 8]
    possible_names = [f"F_D_WM_WW_{num}_F_D_WM_WW_{sample_id_str}.vtk" for num in possible_numbers]
    
    for name in possible_names:
        candidate_path = os.path.join(original_vtk_path, name)
        if os.path.exists(candidate_path):
            original_vtk_file = candidate_path
            break
    
    # If not found, search for any file matching the pattern
    if original_vtk_file is None:
        import glob
        pattern = os.path.join(original_vtk_path, f"F_D_WM_WW_*_F_D_WM_WW_{sample_id_str}.vtk")
        matching_files = glob.glob(pattern)
        if matching_files:
            original_vtk_file = matching_files[0]
            print(f"Found VTK file: {os.path.basename(original_vtk_file)}")
    
    if original_vtk_file is None:
        raise FileNotFoundError(
            f"Original VTK file not found for sample_id {sample_id}. "
            f"Tried patterns: {possible_names} and glob: F_D_WM_WW_*_F_D_WM_WW_{sample_id_str}.vtk"
        )
    
    print(f"Reading VTK file from {original_vtk_file}")
    with suppress_stderr():
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(original_vtk_file)
        reader.Update()
        vtk_data = reader.GetOutput()
    
    num_points = vtk_data.GetNumberOfPoints()
    print(f"VTK file has {num_points} points")
    
    # Verify that number of points matches
    if num_points != num_nodes:
        raise ValueError(
            f"Shape mismatch: Number of points in VTK ({num_points}) does not match "
            f"number of nodes in processed data ({num_nodes}). "
            f"This indicates a mismatch between the VTK file used for processing and the one being read."
        )
    
    # Verify all arrays have the correct length
    expected_len = num_nodes
    if (len(prediction_denorm) != expected_len or 
        len(exact_target_denorm) != expected_len or 
        len(error) != expected_len):
        raise ValueError(
            f"Array length mismatch: All arrays should have length {expected_len}, but got: "
            f"prediction={len(prediction_denorm)}, exact={len(exact_target_denorm)}, "
            f"error={len(error)}"
        )
    
    # 7) Assign nodal features to VTK data
    prediction_vtk = numpy_support.numpy_to_vtk(prediction_denorm, deep=True)
    prediction_vtk.SetName("ML_pred_pressure_Cp")
    
    exact_vtk = numpy_support.numpy_to_vtk(exact_target_denorm, deep=True)
    exact_vtk.SetName("static_pressure_Cp")
    
    error_vtk = numpy_support.numpy_to_vtk(error, deep=True)
    error_vtk.SetName("pressure_error")
    
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
    point_data.SetActiveScalars(prediction_vtk.GetName())
    
    print(f"Added nodal features: {prediction_vtk.GetName()}, {exact_vtk.GetName()}, {error_vtk.GetName()}")
    if node_labels is not None:
        print(f"  - node_labels (geometry token labels)")
    
    # 8) Save VTK file
    output_file = os.path.join(results_path, f"{sample_id:04d}_{model_flag}.vtk")
    print(f"Saving VTK file to {output_file}")
    
    with suppress_stderr():
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(vtk_data)
        writer.SetFileTypeToBinary()  # Use binary format
        writer.Write()
    
    print(f"✅ Successfully saved VTK file: {output_file}")
    print(f"   - Prediction range: [{prediction_denorm.min():.6f}, {prediction_denorm.max():.6f}]")
    print(f"   - Exact range: [{exact_target_denorm.min():.6f}, {exact_target_denorm.max():.6f}]")
    print(f"   - Error range: [{error.min():.6f}, {error.max():.6f}]")
    if node_labels is not None:
        print(f"   - Node labels: {len(np.unique(node_labels))} unique labels")


if __name__ == "__main__":
    # Get inputs from command line or use defaults
    sample_id = 46
    model_name = 'transolver'
    predicted_feature_name = 'pressure'
    
    # Allow override from command line
    if len(sys.argv) > 1:
        sample_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if len(sys.argv) > 3:
        predicted_feature_name = sys.argv[3]
    
    print(f"Processing sample ID: {sample_id}, Model: {model_name}, Feature: {predicted_feature_name}")
    compute_error_and_save_vtk(sample_id, model_name, predicted_feature_name)

