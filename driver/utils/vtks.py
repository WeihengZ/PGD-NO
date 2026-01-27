import vtk
from vtk.util import numpy_support
import torch
import os
import numpy as np
import pyvista as pv

def save_predicted_features_to_vtk(predicted_feature_name,
    original_vtk_path, predicted_features, 
    output_path, normalization_scalars, sim_ground_truth, node_labels=None):
    """
    Save predicted pressure (drag coefficient) to a VTK file using the original mesh geometry.
    Updates the original pressure data to drag coefficient values for consistency.
    
    Args:
        original_vtk_path (str): Path to the original VTK file
        predicted_features (torch.Tensor): Denormalized predicted pressure values (drag coefficient)
        output_path (str): Path to save the output VTK file
        normalization_scalars (dict): Normalization scalars for reference
    """
    # Read the original VTK file to get the mesh geometry
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(original_vtk_path)
    reader.Update()
    
    # Get the polydata
    polydata = reader.GetOutput()
    
    # Create VTK pressure array for predicted values
    predicted_features_np = predicted_features.cpu().numpy()
    pred_features_array = vtk.vtkFloatArray()
    if predicted_feature_name == 'pressure':
        pred_features_array.SetName("ML_pred_pressure_Cp")
    elif predicted_feature_name == 'x_force':
        pred_features_array.SetName("ML_pred_x_force")
    else:
        pred_features_array.SetName(f"ML_pred_{predicted_feature_name}")
    for pressure in predicted_features_np:
        pred_features_array.InsertNextValue(pressure[0])
    
    # Create VTK pressure array for ground truth values
    sim_ground_truth_np = sim_ground_truth.cpu().numpy()
    gt_features_array = vtk.vtkFloatArray()
    if predicted_feature_name == 'pressure':
        gt_features_array.SetName("static_pressure_Cp")
    elif predicted_feature_name == 'x_force':
        gt_features_array.SetName("x_force_gt")
    else:
        gt_features_array.SetName(f"{predicted_feature_name}_gt")
    for pressure in sim_ground_truth_np:
        gt_features_array.InsertNextValue(pressure[0])
    
    # Get the original pressure data for comparison
    point_data = polydata.GetPointData()
        
    # Create error array (difference between predicted and original)
    error_array = vtk.vtkFloatArray()
    error_array.SetName(f"{predicted_feature_name}_error")
    
    for i in range(predicted_features_np.shape[0]):
        if i < len(sim_ground_truth_np):  # Use the converted drag coefficient values
            original_val = sim_ground_truth_np[i]
            pred_val = predicted_features_np[i][0]
            error = np.abs(pred_val - original_val)
            error_array.InsertNextValue(error)
        else:
            error_array.InsertNextValue(0.0)
            
    polydata.GetPointData().AddArray(gt_features_array)
    polydata.GetPointData().AddArray(pred_features_array)
    polydata.GetPointData().AddArray(error_array)
    
    # Add node labels if provided
    if node_labels is not None:
        node_labels_array = vtk.vtkFloatArray()
        node_labels_array.SetName("node_labels")
        for label in node_labels:
            node_labels_array.InsertNextValue(label)
        polydata.GetPointData().AddArray(node_labels_array)
    
    # Write to VTK file - use legacy format for better ParaView compatibility
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()  # Use binary format
    writer.Write()
    
    print(f"Saved predicted {predicted_feature_name} to: {output_path}")

def save_all_predictions_to_vtk(predicted_feature_name, data_dict, test_loader, model, normalization_scalars, 
                               output_folder="processed_vtks", model_name="model", data_folder="data"):
    """
    Save denormalized predicted pressure (drag coefficient) for all test samples to VTK files.
    Updates original pressure data to drag coefficient values for consistency.
    
    Args:
        data_dict (dict): Dictionary containing simulation data
        test_loader: Test data loader
        model: Trained model
        normalization_scalars (dict): Normalization scalars
        output_folder (str): Folder to save VTK files
        model_name (str): Name of the model for file naming
        data_folder (str): Folder containing original VTK files
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create output directory
    model_flag = model_name + '_' + predicted_feature_name
    output_folder = os.path.join(output_folder, model_flag)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n💾 Saving predicted {predicted_feature_name} to VTK files in '{output_folder}'...")
    
    # Store predictions for each simulation
    sim_predictions = {}
    sim_ground_truth = {}
    sim_node_labels = {}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Expected order: features_6d, node_cluster_flags, target, seg_matrix, coef_Cp, integrated_cp_actual, sim_ids
            features_6d, node_cluster_flags, target, seg_matrix, coef_Cp, integrated_cp_actual, sim_ids = batch_data
            features_6d, target = features_6d.to(device), target.to(device)
            coef_Cp = coef_Cp.to(device)
            integrated_cp_actual = integrated_cp_actual.to(device)

            # comput the node label based on the seg matrix
            node_label = np.zeros((features_6d.shape[1]))   # number of nodes
            
            # Convert tensor to numpy (now always dense)
            seg_matrix_np = seg_matrix.detach().cpu().numpy().squeeze()
            
            random_label = np.arange(1, seg_matrix_np.shape[0] + 1)
            np.random.shuffle(random_label)
            for i in range(seg_matrix_np.shape[0]):
                # find the index of the non-zero elements
                non_zero_indices = np.nonzero(seg_matrix_np[i])[0]
                node_label[non_zero_indices] = random_label[i]
            
            # Get predictions
            if model_name == 'transolver_seg':
                seg_matrix = seg_matrix.to(device)
                predicted_field, predicted_integrated_cp = model(features_6d, seg_matrix, coef_Cp)
            elif model_name == 'figconv':
                predicted_field, predicted_integrated_cp = model(features_6d, coef_Cp, node_cluster_flags)
            elif model_name == 'multi':
                predicted_field, predicted_integrated_cp = model(features_6d, coef_Cp, node_cluster_flags)
            else:
                predicted_field, predicted_integrated_cp = model(features_6d, coef_Cp)
            
            # Denormalize according to feature
            if predicted_feature_name == 'pressure':
                pressure_min = torch.tensor(normalization_scalars['pressure_min']).to(device)
                pressure_range = torch.tensor(normalization_scalars['pressure_range']).to(device)
                denormalized_pred = predicted_field * pressure_range + pressure_min
                denormalized_gt = target * pressure_range + pressure_min
            elif predicted_feature_name == 'x_force':
                from utils.metric import denormalize_x_force
                denormalized_pred = denormalize_x_force(predicted_field, normalization_scalars)
                denormalized_gt = denormalize_x_force(target, normalization_scalars)
            else:
                # Default: pass through as-if min-max scalars exist under '{name}_min'/'{name}_range'
                feat_min = torch.tensor(normalization_scalars.get(f'{predicted_feature_name}_min')).to(device)
                feat_range = torch.tensor(normalization_scalars.get(f'{predicted_feature_name}_range')).to(device)
                denormalized_pred = predicted_field * feat_range + feat_min
                denormalized_gt = target * feat_range + feat_min

            # # change it to pressure coefficient
            # denormalized_pressure = (denormalized_pressure - 101325) / (0.5 * 1.204 * 38.89 * 38.89)
            
            # Store predictions for each simulation
            SIM_ID = sim_ids[0]
            sim_predictions[SIM_ID]= denormalized_pred
            sim_ground_truth[SIM_ID] = denormalized_gt
            sim_node_labels[SIM_ID] = node_label

    # Save VTK files for each simulation
    for sim_id in sim_predictions.keys():
        # Concatenate all predictions for this simulation
        predictions = sim_predictions[sim_id][0]
        ground_truth = sim_ground_truth[sim_id][0]
        node_labels = sim_node_labels[sim_id]
        
        # Construct original VTK file path
        original_vtk_path = os.path.join(data_folder, f"data-100k-run{sim_id}-Cp-filtered-fixed-normals.vtk")
        
        if os.path.exists(original_vtk_path):
            # Save to VTK file
            output_path = os.path.join(output_folder, f"prediction_sim_{sim_id}.vtk")
            save_predicted_features_to_vtk(
                predicted_feature_name, original_vtk_path, 
                predictions, output_path, normalization_scalars, ground_truth, node_labels)
        else:
            print(f"Warning: Original VTK file not found: {original_vtk_path}")
    
    print(f"✅ All predictions saved to VTK files!")

def compute_drag_forces_for_all_simulations(data_dict, train_loader, test_loader, model, normalization_scalars, 
                                          output_folder="processed_vtks", model_name="model", data_folder="data"):
    """
    Compute drag forces for both predicted and ground truth pressure (drag coefficient) for all simulations.
    
    Returns:
        dict: Dictionary containing drag forces for each simulation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    drag_forces = {
        'train': {'sim_ids': [], 'gt_drag': [], 'pred_drag': []},
        'test': {'sim_ids': [], 'gt_drag': [], 'pred_drag': []}
    }
    
    # Process training data
    print("\n📊 Computing drag forces for training data...")
    train_predictions = {}
    
    with torch.no_grad():
        for batch_data in train_loader:
            if len(batch_data) == 3:
                features_6d, pressure, sim_ids = batch_data
            else:
                features_6d, pressure = batch_data
                sim_ids = [0] * features_6d.size(0)
            
            features_6d, pressure = features_6d.to(device), pressure.to(device)
            predicted_pressure = model(features_6d)
            
            # Store predictions
            for i, sim_id in enumerate(sim_ids):
                if sim_id not in train_predictions:
                    train_predictions[sim_id] = {'features_6d': [], 'gt_pressure': [], 'pred_pressure': []}
                train_predictions[sim_id]['features_6d'].append(features_6d[i])
                train_predictions[sim_id]['gt_pressure'].append(pressure[i])
                train_predictions[sim_id]['pred_pressure'].append(predicted_pressure[i])
    
    # Process test data
    print("📊 Computing drag forces for test data...")
    test_predictions = {}
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                features_6d, pressure, sim_ids = batch_data
            else:
                features_6d, pressure = batch_data
                sim_ids = [0] * features_6d.size(0)
            
            features_6d, pressure = features_6d.to(device), pressure.to(device)
            predicted_pressure = model(features_6d)
            
            # Store predictions
            for i, sim_id in enumerate(sim_ids):
                if sim_id not in test_predictions:
                    test_predictions[sim_id] = {'features_6d': [], 'gt_pressure': [], 'pred_pressure': []}
                test_predictions[sim_id]['features_6d'].append(features_6d[i])
                test_predictions[sim_id]['gt_pressure'].append(pressure[i])
                test_predictions[sim_id]['pred_pressure'].append(predicted_pressure[i])
    
    # Compute drag forces for each simulation
    for sim_id in train_predictions.keys():
        # Concatenate all data for this simulation
        all_features_6d = torch.cat(train_predictions[sim_id]['features_6d'], dim=0)
        all_gt_pressure = torch.cat(train_predictions[sim_id]['gt_pressure'], dim=0)
        all_pred_pressure = torch.cat(train_predictions[sim_id]['pred_pressure'], dim=0)
        
        # Denormalize pressure
        pressure_min = normalization_scalars['pressure_min']
        pressure_range = normalization_scalars['pressure_range']
        gt_pressure_denormalized = all_gt_pressure * pressure_range + pressure_min
        pred_pressure_denormalized = all_pred_pressure * pressure_range + pressure_min
        
        # Compute drag forces (simplified calculation)
        gt_drag = torch.sum(gt_pressure_denormalized).item()
        pred_drag = torch.sum(pred_pressure_denormalized).item()
        
        drag_forces['train']['sim_ids'].append(sim_id)
        drag_forces['train']['gt_drag'].append(gt_drag)
        drag_forces['train']['pred_drag'].append(pred_drag)
    
    for sim_id in test_predictions.keys():
        # Concatenate all data for this simulation
        all_features_6d = torch.cat(test_predictions[sim_id]['features_6d'], dim=0)
        all_gt_pressure = torch.cat(test_predictions[sim_id]['gt_pressure'], dim=0)
        all_pred_pressure = torch.cat(test_predictions[sim_id]['pred_pressure'], dim=0)
        
        # Denormalize pressure
        pressure_min = normalization_scalars['pressure_min']
        pressure_range = normalization_scalars['pressure_range']
        gt_pressure_denormalized = all_gt_pressure * pressure_range + pressure_min
        pred_pressure_denormalized = all_pred_pressure * pressure_range + pressure_min
        
        # Compute drag forces (simplified calculation)
        gt_drag = torch.sum(gt_pressure_denormalized).item()
        pred_drag = torch.sum(pred_pressure_denormalized).item()
        
        drag_forces['test']['sim_ids'].append(sim_id)
        drag_forces['test']['gt_drag'].append(gt_drag)
        drag_forces['test']['pred_drag'].append(pred_drag)
    
    return drag_forces

def plot_drag_force_comparison(drag_forces, output_path="drag_force_comparison.png"):
    """
    Create a scatter plot comparing predicted vs ground truth drag forces.
    
    Args:
        drag_forces (dict): Dictionary containing drag forces for train and test data
        output_path (str): Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Plot training data
    if drag_forces['train']['gt_drag'] and drag_forces['train']['pred_drag']:
        plt.scatter(drag_forces['train']['gt_drag'], drag_forces['train']['pred_drag'], 
                   c='blue', s=100, alpha=0.7, label='Training Data', edgecolors='black')
    
    # Plot test data
    if drag_forces['test']['gt_drag'] and drag_forces['test']['pred_drag']:
        plt.scatter(drag_forces['test']['gt_drag'], drag_forces['test']['pred_drag'], 
                   c='red', s=100, alpha=0.7, label='Test Data', edgecolors='black')
    
    # Add diagonal line (perfect prediction)
    all_gt = drag_forces['train']['gt_drag'] + drag_forces['test']['gt_drag']
    all_pred = drag_forces['train']['pred_drag'] + drag_forces['test']['pred_drag']
    
    if all_gt and all_pred:
        min_val = min(min(all_gt), min(all_pred))
        max_val = max(max(all_gt), max(all_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth Drag Force', fontsize=12)
    plt.ylabel('Predicted Drag Force', fontsize=12)
    plt.title('Drag Force: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    if all_gt and all_pred:
        from sklearn.metrics import r2_score
        r2 = r2_score(all_gt, all_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Drag force comparison plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n📊 Drag Force Summary:")
    if drag_forces['train']['gt_drag']:
        train_gt = np.array(drag_forces['train']['gt_drag'])
        train_pred = np.array(drag_forces['train']['pred_drag'])
        train_error = np.abs(train_pred - train_gt)
        print(f"Training Data:")
        print(f"  - Mean GT Drag: {np.mean(train_gt):.6f}")
        print(f"  - Mean Pred Drag: {np.mean(train_pred):.6f}")
        print(f"  - Mean Absolute Error: {np.mean(train_error):.6f}")
        print(f"  - Relative Error: {np.mean(train_error/np.abs(train_gt))*100:.2f}%")
    
    if drag_forces['test']['gt_drag']:
        test_gt = np.array(drag_forces['test']['gt_drag'])
        test_pred = np.array(drag_forces['test']['pred_drag'])
        test_error = np.abs(test_pred - test_gt)
        print(f"Test Data:")
        print(f"  - Mean GT Drag: {np.mean(test_gt):.6f}")
        print(f"  - Mean Pred Drag: {np.mean(test_pred):.6f}")
        print(f"  - Mean Absolute Error: {np.mean(test_error):.6f}")
        print(f"  - Relative Error: {np.mean(test_error/np.abs(test_gt))*100:.2f}%")
