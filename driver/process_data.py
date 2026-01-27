import numpy as np
import pickle
import os
from pathlib import Path
import vtk
import pandas as pd
from vtk.util import numpy_support
from utils.seg import create_seg_matrix

DATA_FOLDER = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/selected/"
OUTPUT_FILE = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/processed/"

def compute_normal_vector(polydata):
    """
    Compute unit normal vectors for the polydata surface geometry.
    If normals don't exist, they will be computed using VTK's normal computation.
    
    Args:
        polydata: VTK polydata object
        
    Returns:
        numpy.ndarray: Unit normal vectors of shape (N, 3) where N is the number of points
    """
    # Check if normals already exist
    normals_array = polydata.GetPointData().GetNormals()
    
    if normals_array is None:
        # Compute normals if they don't exist
        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(polydata)
        normal_generator.ComputePointNormalsOn()  # Compute normals at points
        normal_generator.ComputeCellNormalsOff()  # Don't compute cell normals
        normal_generator.SplittingOff()  # Don't split sharp edges
        normal_generator.ConsistencyOn()  # Ensure consistent orientation
        normal_generator.AutoOrientNormalsOn()  # Auto-orient normals
        normal_generator.Update()
        
        # Get the polydata with computed normals
        polydata_with_normals = normal_generator.GetOutput()
        normals_array = polydata_with_normals.GetPointData().GetNormals()
    
    # Convert to numpy array
    normals = numpy_support.vtk_to_numpy(normals_array)  # Shape: (N, 3)
    
    # Normalize to unit vectors (ensure they are unit length)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero for zero-length normals
    norms = np.where(norms == 0, 1.0, norms)
    normals_unit = normals / norms
    
    return normals_unit


def read_vtk_file(file_path):
    """
    Read a VTK file and extract point coordinates, static pressure data, and normal vectors.
    
    Args:
        file_path (str): Path to the VTK file
        
    Returns:
        tuple: (coordinates, pressure, normals, surface_forces, polydata) where:
               - coordinates is (N, 3)
               - pressure is (N, 1)
               - normals is (N, 3) - unit normal vectors
               - surface_forces is (N, 3) with columns [surface_x_force, surface_y_force, surface_z_force]
    """
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    
    # Get the polydata
    polydata = reader.GetOutput()
    
    # Extract point coordinates
    points = polydata.GetPoints()
    coords = numpy_support.vtk_to_numpy(points.GetData())  # Shape: (N, 3)
    
    # Extract static pressure data
    point_data = polydata.GetPointData()
    pressure_array = point_data.GetArray("p")
    
    if pressure_array is None:
        raise ValueError(f"Could not find 'p' field in {file_path}")
    
    pressure = numpy_support.vtk_to_numpy(pressure_array)  # Shape: (N,)
    pressure = pressure.reshape(-1, 1)  # Reshape to (N, 1)

    # Compute unit normal vectors
    normals = compute_normal_vector(polydata)  # Shape: (N, 3) - already unit vectors
    
    return coords, pressure, normals, polydata

def compute_global_statistics(output_dir):
    """
    Compute global statistics across all simulations by loading separate pickle files:
    - Min-max values for coordinates (for min-max normalization)
    - Min-max values for pressure (for min-max normalization)
    - Statistics for normal vectors (for verification)
    
    Args:
        output_dir (str): Path to the directory containing pickle files (one per simulation)
        
    Returns:
        dict: Dictionary with min-max values for coordinates and pressure, and normal statistics
    """
    all_coords = []
    all_pressures = []
    all_normals = []
    all_forces = []
    all_integrated_cp = []
    
    # Find all pickle files in the output directory
    output_path = Path(output_dir)
    pickle_files = sorted(output_path.glob("*.pkl"))
    
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {output_dir}")
    
    print(f"Loading {len(pickle_files)} pickle files to compute global statistics...")
    
    # Load all data from separate pickle files
    for pkl_file in pickle_files:
        # Skip the normalization_scalars file
        if pkl_file.name == "normalization_scalars.pkl":
            continue
            
        with open(pkl_file, 'rb') as f:
            sim_data = pickle.load(f)
            
        all_coords.append(sim_data['coor'])
        all_pressures.append(sim_data['pressure'])
        all_normals.append(sim_data['normals'])
        
        # Collect forces if available
        if 'surface_force' in sim_data:
            all_forces.append(sim_data['surface_force'])
        
        # Collect integrated_cp if available
        if 'integrated_cp_actual' in sim_data:
            all_integrated_cp.append(sim_data['integrated_cp_actual'])
    
    # Concatenate all data
    all_coords = np.vstack(all_coords)      # Shape: (total_points, 3)
    all_pressures = np.vstack(all_pressures) # Shape: (total_points, 1)
    all_normals = np.vstack(all_normals)     # Shape: (total_points, 3)
    
    # Compute min-max for coordinates (for min-max normalization)
    coords_min = np.min(all_coords, axis=0)  # Shape: (3,)
    coords_max = np.max(all_coords, axis=0)  # Shape: (3,)
    
    # Compute min-max for pressure (Cp) - actual normalization
    pressure_min = np.min(all_pressures)     # Scalar
    pressure_max = np.max(all_pressures)     # Scalar
    
    # compute the ranges
    coords_range = coords_max - coords_min
    pressure_range = pressure_max - pressure_min

    normalization_scalars = {
        'coords_min': np.expand_dims(coords_min, axis=0),
        'coords_max': np.expand_dims(coords_max, axis=0),
        'coords_range': np.expand_dims(coords_range,  axis=0),
        'pressure_min': np.expand_dims(pressure_min, axis=0),
        'pressure_max': np.expand_dims(pressure_max, axis=0),
        'pressure_range': np.expand_dims(pressure_range, axis=0),
    }
    
    # Compute force statistics if forces exist
    if len(all_forces) > 0:
        all_forces = np.vstack(all_forces)  # Shape: (total_points, 3)
        force_min = np.min(all_forces, axis=0)  # Shape: (3,)
        force_max = np.max(all_forces, axis=0)  # Shape: (3,)
        force_range = force_max - force_min
        normalization_scalars['force_min'] = np.expand_dims(force_min, axis=0)
        normalization_scalars['force_max'] = np.expand_dims(force_max, axis=0)
        normalization_scalars['force_range'] = np.expand_dims(force_range, axis=0)
    else:
        # Set default values if forces don't exist
        normalization_scalars['force_min'] = np.array([[0.0, 0.0, 0.0]])
        normalization_scalars['force_range'] = np.array([[1.0, 1.0, 1.0]])
    
    # Compute integrated_cp statistics if available
    if len(all_integrated_cp) > 0:
        all_integrated_cp = np.array(all_integrated_cp)  # Shape: (num_sims,)
        integrated_cp_min = np.min(all_integrated_cp)
        integrated_cp_max = np.max(all_integrated_cp)
        integrated_cp_range = integrated_cp_max - integrated_cp_min
        normalization_scalars['integrated_cp_min'] = np.expand_dims(integrated_cp_min, axis=0)
        normalization_scalars['integrated_cp_max'] = np.expand_dims(integrated_cp_max, axis=0)
        normalization_scalars['integrated_cp_range'] = np.expand_dims(integrated_cp_range, axis=0)
    
    return normalization_scalars

def normalize_data(output_dir, normalization_scalars):
    """
    Apply min-max normalization for both coordinates and pressure by loading and updating separate pickle files:
    - Min-max normalization for coordinates (scales to [0,1] range)
    - Min-max normalization for pressure (scales to [0,1] range)
    Create 6D features by concatenating normalized coordinates with normal vectors.
    
    Args:
        output_dir (str): Path to the directory containing pickle files (one per simulation)
        normalization_scalars (dict): Dictionary with min-max values for coordinates and pressure
        
    Returns:
        int: Number of files processed
    """
    coords_min = normalization_scalars['coords_min']
    coords_range = normalization_scalars['coords_range']
    pressure_min = normalization_scalars['pressure_min']
    pressure_range = normalization_scalars['pressure_range']
    force_min = normalization_scalars.get('force_min', np.array([[0.0, 0.0, 0.0]]))
    force_range = normalization_scalars.get('force_range', np.array([[1.0, 1.0, 1.0]]))
    integrated_cp_min = normalization_scalars.get('integrated_cp_min', np.array([[0.0]]))
    integrated_cp_range = normalization_scalars.get('integrated_cp_range', np.array([[1.0]]))
    
    # Find all pickle files in the output directory
    output_path = Path(output_dir)
    pickle_files = sorted(output_path.glob("*.pkl"))
    
    num_processed = 0
    
    # Process each simulation file
    for pkl_file in pickle_files:
        # Skip the normalization_scalars file
        if pkl_file.name == "normalization_scalars.pkl":
            continue
        
        # Load the simulation data
        with open(pkl_file, 'rb') as f:
            sim_data = pickle.load(f)
        
        # Do nothing for coordinates
        coords_normalized = sim_data['coor'] 
        
        # Min-max normalization for pressure (Cp): (pressure - min) / (max - min)
        pressure_normalized = (sim_data['pressure'] - pressure_min) / pressure_range

        # Normalize surface forces per component if available
        if 'surface_force' in sim_data:
            forces_normalized = (sim_data['surface_force'] - force_min) / force_range
            sim_data['surface_force'] = forces_normalized
        
        # Create 6D features: [normalized_coords, normals]
        # Note: normals are already unit vectors, so we don't normalize them
        features_6d = np.concatenate([coords_normalized, sim_data['normals']], axis=1)
        
        # Normalize integrated Cp scalar per simulation if present
        if 'integrated_cp_actual' in sim_data:
            sim_data['integrated_cp_actual_normalized'] = (sim_data['integrated_cp_actual'] - integrated_cp_min) / integrated_cp_range

        # Update the data
        sim_data['coor'] = coords_normalized
        sim_data['pressure'] = pressure_normalized
        sim_data['features_6d'] = features_6d  # New 6D feature array
        
        # Save the updated data back to the same file
        with open(pkl_file, 'wb') as f:
            pickle.dump(sim_data, f)
        
        num_processed += 1
    
    return num_processed

def process_data_folder(
    threshold_angles, min_graph_size_ratio, data_folder, 
    output_file=OUTPUT_FILE):
    """
    Process all VTK files in the data folder, standardize the data, and save as a pickle file.
    
    Args:
        data_folder (str): Path to the folder containing VTK files
        output_file (str): Path to save the output pickle file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_file, exist_ok=True)

    # Find all VTK files in the data folder
    data_path = Path(data_folder)
    vtk_files = list(data_path.glob("*.vtk"))
    
    # Dictionary to store all data
    num_processed = 0
    print(f"Found {len(vtk_files)} VTK files to process...")
    for vtk_file in sorted(vtk_files):
        num_processed += 1
        if num_processed > 2:
            break

        print(f"Processing {vtk_file.name}...")
        
        # Extract simulation ID from filename
        filename = vtk_file.stem  # Remove .vtk extension
        sim_id = filename.split("_")[-1]
        print(sim_id)
        sim_id = int(sim_id)

        if os.path.exists(f"{output_file}/{sim_id}.pkl"):
            print(f"Skipping {vtk_file.name} because it already exists...")
            continue
        
        # Read the VTK file
        coords, pressure, normals, polydata = read_vtk_file(str(vtk_file))

        # create the seg matrix
        seg_matrix, node_cluster_flags = create_seg_matrix(coords, polydata, 
            threshold_angles=threshold_angles, 
            min_graph_size_ratio=min_graph_size_ratio,
            FIND_SEG=True,
            MERGE_SMALL_GRAPHS=False,
            INCLUDE_NO_CLUSTER_NODES=False)
        
        # save the node labels as a vtk file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(vtk_file))
        reader.Update()
        polydata = reader.GetOutput()
        
        # # save a vtk of the node_cluster_flags using polydata
        # # assign the node_cluster_flags to the point data of the polydata
        # # save the polydata as a vtk file under the same folder
        # node_labels_array = vtk.vtkFloatArray()
        # node_labels_array.SetName("node_labels")
        
        # for label in node_cluster_flags.flatten():
        #     node_labels_array.InsertNextValue(label)
        
        # polydata.GetPointData().AddArray(node_labels_array)
        
        # # Write to VTK file - use legacy format for better ParaView compatibility
        # writer = vtk.vtkPolyDataWriter()
        # output_path = "./test_vtk.vtk"
        # writer.SetFileName(str(output_path))
        # writer.SetInputData(polydata)
        # writer.SetFileTypeToBinary()  # Use binary format
        # writer.Write()
        # assert False

        # Store in dictionary using sim_id as key
        data_dict = {
            "coor": coords,      # Shape: (M, 3)
            "node_cluster_flags": node_cluster_flags, # Shape: (M,)
            "pressure": pressure,  # Shape: (M, 1)
            "normals": normals,    # Shape: (M, 3)
            "seg_matrix": seg_matrix, # Shape: (S, M)
        }

        # save the data_dict as a pickle file
        with open(f"{output_file}/{sim_id}.pkl", 'wb') as f:
            pickle.dump(data_dict, f)
        
    # Compute global statistics values from all processed files
    print("Computing global statistics values for normalization...")
    normalization_scalars = compute_global_statistics(output_file)
    
    # Save normalization scalars to a separate file
    normalization_file = os.path.join(output_file, "normalization_scalars.pkl")
    with open(normalization_file, 'wb') as f:
        pickle.dump(normalization_scalars, f)
    print(f"Normalization scalars saved to {normalization_file}")
    
    # Apply normalization to all processed files
    print("Applying normalization...")
    num_normalized = normalize_data(output_file, normalization_scalars)
    
    print(f"\nNormalization complete!")
    print(f"Total simulations processed: {num_processed}")
    print(f"Total simulations normalized: {num_normalized}")
    
    return normalization_scalars

def load_processed_data(file_path="processed_data/data_dict.pkl"):
    """
    Load the processed data from pickle file.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        dict: The loaded data dictionary
    """
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

if __name__ == "__main__":
    # Process the data
    normalization_scalars = process_data_folder(
        threshold_angles=np.arange(8, 5, -1), 
        min_graph_size_ratio=0.0001,
        data_folder=DATA_FOLDER)
    
    # Example of how to access the data
    print("\nExample data access:")
    output_path = Path(OUTPUT_FILE)
    pickle_files = sorted(output_path.glob("*.pkl"))
    
    # Load and display first simulation file (skip normalization_scalars)
    for pkl_file in pickle_files:
        if pkl_file.name == "normalization_scalars.pkl":
            continue
        
        with open(pkl_file, 'rb') as f:
            sim_data = pickle.load(f)
        
        sim_id = pkl_file.stem
        print(f"Simulation {sim_id}:")
        print(f"  - Coordinates: {sim_data['coor'].shape}")
        print(f"  - Pressure: {sim_data['pressure'].shape}")
        print(f"  - Normals: {sim_data['normals'].shape}")
        if 'features_6d' in sim_data:
            print(f"  - 6D Features: {sim_data['features_6d'].shape}")
            print(f"  - First 3 normalized coordinates (min-max to [0,1]): {sim_data['coor'][:3]}")
            print(f"  - First 3 normalized pressure values: {sim_data['pressure'][:3].flatten()}")
            print(f"  - First 3 normal vectors: {sim_data['normals'][:3]}")
            print(f"  - First 3 6D features: {sim_data['features_6d'][:3]}")
        break  # Just show first simulation as example