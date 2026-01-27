import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


# Path to the JEB data folder containing separate pkl files
DATA_PATH = "./"
NORMALIZATION_PATH = os.path.join(DATA_PATH, "mean_std.npz")


def custom_collate_fn(batch):
    """
    Custom collate function for JEB data.

    Expects each item from the dataset to be a 4-tuple:
        (vertices, stress, surface_nodes, seg_matrix)

    Returns:
        vertices_batch:      (B, N, 3) float32 tensor
        stress_batch:        (B, N)   or (B, N, 1) float32 tensor
        surface_nodes_batch: (B, M, 3) float32 tensor
        seg_matrix_batch:    (B, N_token, M) float32 tensor
    """
    first = batch[0]
    n_fields = len(first)

    if n_fields != 4:
        raise ValueError(
            f"Unexpected number of fields in batch elements: {n_fields}. "
            "Expected 4-tuple: (vertices, stress, surface_nodes, seg_matrix)."
        )

    vertices_list, stress_list, surface_nodes_list, seg_matrix_list = zip(*batch)

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

    vertices_batch = torch.stack([_to_tensor(x) for x in vertices_list], dim=0)
    stress_batch = torch.stack([_to_tensor(x) for x in stress_list], dim=0)
    surface_nodes_batch = torch.stack([_to_tensor(x) for x in surface_nodes_list], dim=0)
    seg_matrix_batch = torch.stack([_to_tensor(x) for x in seg_matrix_list], dim=0)

    return vertices_batch, stress_batch, surface_nodes_batch, seg_matrix_batch


class JEBDataset(Dataset):
    """
    Dataset for JEB where each sample is stored in a separate .pkl file.

    Expected structure for each .pkl file:
        {
            "volume_nodes":      (N, 3),
            "volume_features":   (N,) or (N, 1)  # e.g., stress or scalar field
            "surface_features": (M, 6),
            "seg_matrix":    (N_token, M) or csr_matrix,
        }
    
    Files are loaded on-the-fly when __getitem__ is called.
    """

    def __init__(self, data_folder: str, indices: List[str]):
        """
        Args:
            data_folder: Path to folder containing separate .pkl files
            indices: List of file names (without .pkl extension) or full file names to use
        """
        self.data_folder = data_folder
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        file_name = self.indices[idx]
        
        # Ensure file_name has .pkl extension
        if not file_name.endswith('.pkl'):
            file_name = file_name + '.pkl'
        
        file_path = os.path.join(self.data_folder, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JEB data file not found: {file_path}")

        # Load the pkl file on-the-fly
        with open(file_path, "rb") as f:
            sample = pickle.load(f)

        if not isinstance(sample, dict):
            raise TypeError(f"Expected dict from {file_path}, got {type(sample)}")

        # Required fields
        try:
            vertices = sample["volume_nodes"]
            stress = sample["volume_features"]
            surface_nodes = sample["surface_features"]
            seg_matrix = sample["seg_matrix"]
        except KeyError as e:
            raise KeyError(f"Missing key {e} in JEB sample {file_path}") from e

        # Convert seg_matrix if sparse
        if isinstance(seg_matrix, csr_matrix):
            seg_matrix = seg_matrix.toarray()

        return vertices, stress, surface_nodes, seg_matrix


def discover_indices(data_folder: str = DATA_PATH) -> List[str]:
    """
    Discover all available .pkl files in the data folder.
    
    Args:
        data_folder: Path to folder containing separate .pkl files
        
    Returns:
        Sorted list of file names (without .pkl extension)
    """
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"JEB data folder not found: {data_folder}")
    
    # Find all .pkl files in the folder
    pkl_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
    
    # Remove .pkl extension and sort
    indices = sorted([os.path.splitext(f)[0] for f in pkl_files])
    
    return indices


def create_data_loaders(
    data_folder: str = DATA_PATH,
    batch_size: int = 1,
    train_index: List[str] = None,
    val_index: List[str] = None,
    test_index: List[str] = None,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders for the JEB dataset.

    Args:
        data_folder: Path to folder containing separate .pkl files
        batch_size:  Batch size for all loaders.
        train_index: List of file names (without .pkl extension) for training.
        val_index:   List of file names (without .pkl extension) for validation.
        test_index:  List of file names (without .pkl extension) for testing.
        shuffle:     Whether to shuffle the training loader.
        num_workers: Number of worker processes for data loading.
        pin_memory:  Whether to pin memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    all_indices = discover_indices(data_folder)
    print(f"Available JEB sample files: {len(all_indices)} total")

    if train_index is None and val_index is None and test_index is None:
        # Default: all samples for training
        train_indices = all_indices
        val_indices = []
        test_indices = []
    else:
        train_indices = train_index or []
        val_indices = val_index or []
        test_indices = test_index or []

    print("JEB data split:")
    print(f"  - Training:   {len(train_indices)} simulations")
    print(f"  - Validation: {len(val_indices)} simulations")
    print(f"  - Testing:    {len(test_indices)} simulations")

    train_dataset = JEBDataset(data_folder, train_indices)
    val_dataset = JEBDataset(data_folder, val_indices) if val_indices else JEBDataset(data_folder, [])
    test_dataset = JEBDataset(data_folder, test_indices) if test_indices else JEBDataset(data_folder, [])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test loading from separate files
    all_indices = discover_indices(DATA_PATH)
    print(f"Found {len(all_indices)} .pkl files in {DATA_PATH}")
    
    # Get a few sample file names for testing
    sample_keys = all_indices[:3] if len(all_indices) >= 3 else all_indices
    print(f"Testing with samples: {sample_keys}")

    train_loader, val_loader, test_loader = create_data_loaders(
        DATA_PATH, batch_size=1, train_index=sample_keys, val_index=sample_keys, test_index=sample_keys
    )

    print(f"\nDataLoader test (JEB):")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches:   {len(val_loader)}")
    print(f"  - Test batches:  {len(test_loader)}")

    for batch in train_loader:
        vertices, stress, surface_nodes, seg_matrix = batch
        print("vertices shape:", vertices.shape)
        print("stress shape:", stress.shape)
        print("surface_nodes shape:", surface_nodes.shape)
        print("seg_matrix shape:", seg_matrix.shape)
        break

