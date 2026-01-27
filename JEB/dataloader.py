import os
import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


# Path to the consolidated JEB data file
DATA_PATH = "./"
ALL_JEB_DATA_PATH = os.path.join(DATA_PATH, "all_jeb.pkl")
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
    Dataset for JEB where all samples are stored in a single consolidated .pkl file.

    Expected structure for all_jeb.pkl:
        {
            "sim_id_1": {
                "volume_nodes":      (N, 3),
                "volume_features":   (N,) or (N, 1)  # e.g., stress or scalar field
                "surface_nodes": (M, 3),
                "seg_matrix":    (N_token, M) or csr_matrix,
            },
            "sim_id_2": { ... },
            ...
        }
    """

    def __init__(self, data_dict: dict, indices: List[str]):
        """
        Args:
            data_dict: Dictionary loaded from all_jeb.pkl
            indices: List of sim_id strings to use
        """
        self.data_dict = data_dict
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sim_id = self.indices[idx]
        
        if sim_id not in self.data_dict:
            raise KeyError(f"Sample {sim_id} not found in JEB data dictionary")

        sample = self.data_dict[sim_id]

        # Required fields
        try:
            vertices = sample["volume_nodes"]
            stress = sample["volume_features"]
            surface_nodes = sample["surface_nodes"]
            seg_matrix = sample["seg_matrix"]
        except KeyError as e:
            raise KeyError(f"Missing key {e} in JEB sample {sim_id}") from e

        # # Convert seg_matrix if sparse
        # if isinstance(seg_matrix, csr_matrix):
        #     seg_matrix = seg_matrix.toarray()

        return vertices, stress, surface_nodes, seg_matrix


def load_data(data_path: str = ALL_JEB_DATA_PATH) -> dict:
    """
    Load the consolidated JEB data from all_jeb.pkl.
    
    Args:
        data_path: Path to all_jeb.pkl file
        
    Returns:
        Dictionary with sim_id keys and sample dicts as values
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"JEB consolidated data file not found: {data_path}")
    
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    if not isinstance(data_dict, dict):
        raise TypeError(f"Expected dict from {data_path}, got {type(data_dict)}")
    
    return data_dict


def discover_indices(data_dict: dict = None) -> List[str]:
    """
    Discover all available sample indices from the consolidated data dictionary.
    
    Args:
        data_dict: Dictionary loaded from all_jeb.pkl. If None, loads it.
        
    Returns:
        Sorted list of sim_id strings
    """
    if data_dict is None:
        data_dict = load_data(ALL_JEB_DATA_PATH)
    
    indices = sorted(list(data_dict.keys()))
    return indices


def create_data_loaders(
    data_source: Union[str, dict] = ALL_JEB_DATA_PATH,
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
        data_source: Path to all_jeb.pkl file, or a pre-loaded dict.
        batch_size:  Batch size for all loaders.
        train_index: List of sim_id strings for training.
        val_index:   List of sim_id strings for validation.
        test_index:  List of sim_id strings for testing.
        shuffle:     Whether to shuffle the training loader.
        num_workers: Number of worker processes for data loading.
        pin_memory:  Whether to pin memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load data if path provided
    if isinstance(data_source, str):
        data_dict = load_data(data_source)
    else:
        data_dict = data_source
    
    all_indices = discover_indices(data_dict)
    print(f"Available JEB sample indices: {len(all_indices)} total")

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

    train_dataset = JEBDataset(data_dict, train_indices)
    val_dataset = JEBDataset(data_dict, val_indices) if val_indices else JEBDataset(data_dict, [])
    test_dataset = JEBDataset(data_dict, test_indices) if test_indices else JEBDataset(data_dict, [])

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
    # Test loading from consolidated file
    data_dict = load_data(ALL_JEB_DATA_PATH)
    print(f"Loaded {len(data_dict)} samples from {ALL_JEB_DATA_PATH}")
    
    # Get a sample key for testing
    sample_keys = list(data_dict.keys())[:3]
    print(f"Testing with samples: {sample_keys}")

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict, batch_size=1, train_index=sample_keys, val_index=sample_keys, test_index=sample_keys
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

