import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


# Path to the processed heatsink dataset (single pickle file)
DATAPATH = "./processed_data.pkl"


def custom_collate_fn(batch):
    """
    Custom collate function for heatsink data.

    Expects each item from the dataset to be a 4-tuple:
        (vertices, temperatures, surface_nodes, seg_matrix)

    Returns:
        vertices_batch:      (B, N, 3) float32 tensor
        temperatures_batch:  (B, N)   float32 tensor
        surface_nodes_batch: (B, M, 3) float32 tensor
        seg_matrix_batch:    (B, N_token, M) float32 tensor
    """
    first = batch[0]
    n_fields = len(first)

    if n_fields != 4:
        raise ValueError(
            f"Unexpected number of fields in batch elements: {n_fields}. "
            "Expected 4-tuple: (vertices, temperatures, surface_nodes, seg_matrix)."
        )

    vertices_list, temperatures_list, surface_nodes_list, seg_matrix_list = zip(*batch)

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

    vertices_batch = torch.stack([_to_tensor(x) for x in vertices_list], dim=0)
    temperatures_batch = torch.stack([_to_tensor(x) for x in temperatures_list], dim=0)
    surface_nodes_batch = torch.stack(
        [_to_tensor(x) for x in surface_nodes_list], dim=0
    )
    seg_matrix_batch = torch.stack([_to_tensor(x) for x in seg_matrix_list], dim=0)

    return vertices_batch, temperatures_batch, surface_nodes_batch, seg_matrix_batch


class HeatSinkDataset(Dataset):
    """
    Dataset for the heatsink processed data.

    Assumes the loaded pickle at DATAPATH is a dict with field-wise lists:
        {
            "vertices":       [sample_0_vertices, sample_1_vertices, ...],
            "temperatures":   [sample_0_t,        sample_1_t,        ...],
            "surface_nodes":  [sample_0_nodes,    sample_1_nodes,    ...],
            "seg_matrix":     [sample_0_seg,      sample_1_seg,      ...],
        }
    All lists must have the same length (number of samples).

    For each sample index i, this dataset returns:
        vertices_i      = vertices[i]
        temperatures_i  = temperatures[i]
        surface_nodes_i = surface_nodes[i]
        seg_mat_i       = seg_matrix[i]
        (no explicit sim_id is returned from __getitem__)
    """

    def __init__(self, data_dict: Dict, indices: List[int]):
        # Basic sanity checks on expected keys
        required_keys = ["vertices", "temperatures", "surface_nodes", "seg_matrix"]
        for k in required_keys:
            if k not in data_dict:
                raise KeyError(f"Heatsink data_dict missing required key: '{k}'")

        n_vertices = len(data_dict["vertices"])
        n_temps = len(data_dict["temperatures"])
        n_surface = len(data_dict["surface_nodes"])
        n_seg = len(data_dict["seg_matrix"])

        if not (n_vertices == n_temps == n_surface == n_seg):
            raise ValueError(
                "Heatsink data_dict lists must all share the same length, got: "
                f"vertices={n_vertices}, temperatures={n_temps}, "
                f"surface_nodes={n_surface}, seg_matrix={n_seg}"
            )

        self.data_dict = data_dict
        self.indices = indices
        self.num_samples = n_vertices

        # Optional: validate that indices are in range
        for idx in indices:
            if idx < 0 or idx >= self.num_samples:
                raise IndexError(
                    f"Sample index {idx} out of range [0, {self.num_samples - 1}]"
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sim_id = self.indices[idx]

        vertices = self.data_dict["vertices"][sim_id]
        temperatures = self.data_dict["temperatures"][sim_id]
        surface_nodes = self.data_dict["surface_nodes"][sim_id]
        seg_matrix = self.data_dict["seg_matrix"][sim_id]

        # Convert seg_matrix if sparse
        if isinstance(seg_matrix, csr_matrix):
            seg_matrix = seg_matrix.toarray()

        return vertices, temperatures, surface_nodes, seg_matrix


def load_data(data_path: str = DATAPATH) -> Dict:
    """
    Load the processed heatsink data from a single pickle file.

    Args:
        data_path: Path to the processed_data.pkl file.

    Returns:
        data_dict: Dictionary holding all simulations.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Heatsink data file not found: {data_path}")

    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    if not isinstance(data_dict, dict):
        raise TypeError(
            f"Loaded heatsink data must be a dict, got {type(data_dict)} instead."
        )

    return data_dict


def create_data_loaders(
    data_source: Union[Dict, str] = DATAPATH,
    batch_size: int = 1,
    train_index: List[int] = None,
    val_index: List[int] = None,
    test_index: List[int] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders for the heatsink dataset.

    Args:
        data_source: Either a dict already loaded from pickle, or a path string
                     to the heatsink processed_data.pkl.
        batch_size:  Batch size for all loaders.
        train_index: List of indices for training.
        val_index:   List of indices for validation.
        test_index:  List of indices for testing.
        shuffle:     Whether to shuffle the training loader.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load from path if needed
    if isinstance(data_source, str):
        data_dict = load_data(data_source)
    else:
        data_dict = data_source

    # All available sample indices come from the length of the 'vertices' list
    if "vertices" not in data_dict:
        raise KeyError("Heatsink data_dict must contain key 'vertices'")

    num_samples = len(data_dict["vertices"])
    all_indices = list(range(num_samples))
    print(f"Available heatsink sample indices: {all_indices}")

    # Default split: use all data for training if no indices are provided
    if train_index is None and val_index is None and test_index is None:
        train_indices = all_indices
        val_indices = []
        test_indices = []
    else:
        train_indices = train_index or []
        val_indices = val_index or []
        test_indices = test_index or []

    print("Heatsink data split:")
    print(f"  - Training:   {len(train_indices)} simulations (indices: {train_indices})")
    print(f"  - Validation: {len(val_indices)} simulations (indices: {val_indices})")
    print(f"  - Testing:    {len(test_indices)} simulations (indices: {test_indices})")

    # Build datasets (empty splits allowed)
    train_dataset = HeatSinkDataset(data_dict, train_indices)
    val_dataset = HeatSinkDataset(data_dict, val_indices) if val_indices else HeatSinkDataset(data_dict, [])
    test_dataset = HeatSinkDataset(data_dict, test_indices) if test_indices else HeatSinkDataset(data_dict, [])

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Simple smoke test: load heatsink data and print one batch.
    """
    data_dict = load_data(DATAPATH)

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict, batch_size=1, train_index=[0], val_index=[0], test_index=[0]
    )

    print(f"\nDataLoader test (heatsink):")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches:   {len(val_loader)}")
    print(f"  - Test batches:  {len(test_loader)}")

    for batch in train_loader:
        vertices, temperatures, surface_nodes, seg_matrix = batch
        print("vertices shape:", vertices.shape)
        print("temperatures shape:", temperatures.shape)
        print("surface_nodes shape:", surface_nodes.shape)
        print("seg_matrix shape:", seg_matrix.shape)
        break