import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix


def custom_collate_fn(batch):
    """
    Custom collate function for batch processing.
    Expects 4-tuples from `PerFileVTKDataset`:
        (coorf, seg_matrix, p, sim_id)
    """
    first = batch[0]
    n_fields = len(first)

    if n_fields != 4:
        raise ValueError(
            f"Unexpected number of fields in batch elements: {n_fields}. "
            "Expected 4-tuple: (coorf, seg_matrix, p, sim_id)."
        )

    coorf_list, seg_matrix_list, p_list, sim_id = zip(*batch)

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

    coorf_batch = torch.stack([_to_tensor(x) for x in coorf_list], dim=0)
    seg_matrix_batch = torch.stack([_to_tensor(x) for x in seg_matrix_list], dim=0)
    p_batch = torch.stack([_to_tensor(x) for x in p_list], dim=0)
    sim_id_batch = list(sim_id)

    return coorf_batch, seg_matrix_batch, p_batch, sim_id_batch


class _NumpyCoreAliasUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps modules from environments that saved pickles with
    old/private NumPy package paths (e.g., 'numpy._core') to current paths
    (e.g., 'numpy.core'). This helps load pickles across NumPy versions.
    """

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        # Add other common aliases here if encountered in the future
        # "numpy._multiarray_umath": "numpy.core._multiarray_umath",
    }

    def find_class(self, module, name):
        remapped_module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(remapped_module, name)


class VTKDataset(Dataset):
    """
    Custom Dataset for VTK data with 6D features (coordinates + normal vectors)
    and pressure or x-force, when everything is already loaded in a single
    in-memory dictionary (legacy mode).
    """

    def __init__(self, data_dict, indices, predicted_feature_name="pressure"):
        self.data = []
        self.indices = indices
        self.sim_ids = []  # Track simulation IDs
        self.predicted_feature_name = predicted_feature_name

        # Extract data for specified indices
        for idx in indices:
            if idx in data_dict and idx != "normalization_scalars":
                # Use 6D features (min-max normalized coords + normal vectors)
                features_6d = torch.FloatTensor(data_dict[idx]["features_6d"])
                node_cluster_flags = data_dict[idx]["node_cluster_flags"]

                if "seg_matrix" in data_dict[idx].keys():
                    # Check if seg_matrix is already a sparse matrix
                    if isinstance(data_dict[idx]["seg_matrix"], csr_matrix):
                        # Convert sparse matrix to dense tensor
                        seg_matrix = torch.FloatTensor(
                            data_dict[idx]["seg_matrix"].toarray()
                        )
                    else:
                        # Convert dense matrix to tensor
                        seg_matrix = torch.FloatTensor(data_dict[idx]["seg_matrix"])
                else:
                    # Create empty tensor if no seg_matrix
                    seg_matrix = torch.FloatTensor(np.zeros((1, 1, 1)))

                # extract predicted features
                pressure = torch.FloatTensor(data_dict[idx]["pressure"])
                force = torch.FloatTensor(data_dict[idx]["surface_force"])

                # Extract target based on predicted_feature_name
                if self.predicted_feature_name == "x_force":
                    # Extract first dimension (x-component) of force
                    target = force[:, 0:1]  # Keep dimension as (N, 1) for consistency
                elif self.predicted_feature_name == "y_force":
                    # Extract second dimension (y-component) of force
                    target = force[:, 1:2]  # Keep dimension as (N, 1) for consistency
                elif self.predicted_feature_name == "z_force":
                    # Extract third dimension (z-component) of force
                    target = force[:, 2:3]  # Keep dimension as (N, 1) for consistency
                else:  # default to pressure
                    target = pressure

                coef_Cp = torch.FloatTensor(data_dict[idx]["integrate_coef"])
                integrated_cp_actual = torch.FloatTensor(
                    np.array([data_dict[idx]["integrated_cp_actual"]])
                )

                # append the data
                self.data.append(
                    (
                        features_6d,
                        node_cluster_flags,
                        target,
                        seg_matrix,
                        coef_Cp,
                        integrated_cp_actual,
                    )
                )
                self.sim_ids.append(idx)  # Store the simulation ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            features_6d,
            node_cluster_flags,
            target,
            seg_matrix,
            coef_Cp,
            integrated_cp_actual,
        ) = self.data[idx]
        sim_id = self.sim_ids[idx]

        return (
            features_6d,
            node_cluster_flags,
            target,
            seg_matrix,
            coef_Cp,
            integrated_cp_actual,
            sim_id,
        )


class PerFileVTKDataset(Dataset):
    """
    Dataset variant that loads **one simulation per file** on demand.

    Expected layout (can be adjusted if your naming is different):
        data_root/
            1.pkl
            2.pkl
            ...

    Each per-index pickle is expected to contain the same fields that were
    previously stored in `data_dict[idx]`.
    """

    def __init__(self, data_root, indices, predicted_feature_name="pressure"):
        self.data_root = data_root
        self.indices = indices
        self.predicted_feature_name = predicted_feature_name

    def _sample_path(self, sim_id):
        # If your naming is different (e.g. f"sample_{sim_id}.pkl"),
        # we can tweak this in the next step.
        return os.path.join(self.data_root, f"{sim_id}.pkl")

    def _load_sample(self, sim_id):
        file_path = self._sample_path(sim_id)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Per-sample file not found: {file_path}")

        with open(file_path, "rb") as f:
            sample = pickle.load(f)
        return sample

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map from dataset index to simulation id
        sim_id = self.indices[idx]
        sample = self._load_sample(sim_id)
        coorf = sample["features_6d"]

        # Ensure seg_matrix is dense (mirror behavior of VTKDataset)
        seg_matrix = sample["seg_matrix"]
        if isinstance(seg_matrix, csr_matrix):
            seg_matrix = seg_matrix.toarray()

        p = sample["pressure"]

        return (coorf, seg_matrix, p, sim_id)



def create_data_loaders(
    data_source,
    batch_size=1,
    train_index=[66],
    val_index=[66],
    test_index=[66],
    shuffle=True,
    predicted_feature_name="pressure",
):
    """
    Create train, validation, and test data loaders based on specific indices.

    There are two modes:
      1. Legacy in-memory mode: `data_source` is a dict like the old `data_dict`.
      2. Per-file lazy mode:  `data_source` is a directory path. Each sample
         with index `i` is loaded from a file like `<data_source>/<i>.pkl`.

    Args:
        data_source (dict | str): Either the in-memory data_dict, or the
            directory containing one file per sample.
        batch_size (int): Batch size for data loaders
        train_index (list): List of indices for training data
        val_index (list): List of indices for validation data
        test_index (list): List of indices for testing data
        shuffle (bool): Whether to shuffle training data
        predicted_feature_name (str): Name of the feature to predict
            ("pressure", "x_force", "y_force", "z_force")

    Returns:
        tuple: (train_loader, val_loader, test_loader, normalization_scalars)
            In per-file mode, `normalization_scalars` is returned as None
            (we can wire this up in a later step if needed).
    """

    is_dict_mode = isinstance(data_source, dict)
    is_path_mode = isinstance(data_source, str)

    if not (is_dict_mode or is_path_mode):
        raise TypeError(
            f"data_source must be either a dict or a str (path), got {type(data_source)}"
        )

    # Use provided indices
    train_indices = train_index
    val_indices = val_index
    test_indices = test_index

    if is_dict_mode:
        data_dict = data_source
        # Get simulation indices (excluding normalization_scalars)
        sim_indices = [idx for idx in data_dict.keys() if idx != "normalization_scalars"]
        sim_indices.sort()
        print(f"Available simulation indices: {sim_indices}")

        normalization_scalars = data_dict.get("normalization_scalars", None)

        # Create datasets with predicted_feature_name (legacy in-memory)
        train_dataset = VTKDataset(
            data_dict, train_indices, predicted_feature_name=predicted_feature_name
        )
        val_dataset = VTKDataset(
            data_dict, val_indices, predicted_feature_name=predicted_feature_name
        )
        test_dataset = VTKDataset(
            data_dict, test_indices, predicted_feature_name=predicted_feature_name
        )

    else:
        data_root = data_source
        print(f"Per-file data root: {data_root}")
        # We don't know normalization_scalars location yet in per-file mode.
        normalization_scalars = None

        # Create datasets that load one file per sample on demand
        train_dataset = PerFileVTKDataset(
            data_root, train_indices, predicted_feature_name=predicted_feature_name
        )
        val_dataset = PerFileVTKDataset(
            data_root, val_indices, predicted_feature_name=predicted_feature_name
        )
        test_dataset = PerFileVTKDataset(
            data_root, test_indices, predicted_feature_name=predicted_feature_name
        )

    print("Data split:")
    print(f"  - Training: {len(train_indices)} simulations (indices: {train_indices})")
    print(f"  - Validation: {len(val_indices)} simulations (indices: {val_indices})")
    print(f"  - Testing: {len(test_indices)} simulations (indices: {test_indices})")
    print(f"  - Predicting: {predicted_feature_name}")

    # Create data loaders with custom collate function for sparse tensors
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, normalization_scalars


if __name__ == "__main__":

    DATA_PATH = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/processed/"
    with open(DATA_PATH + "normalization_scalars.pkl", "rb") as f:
        normalization_scalars = pickle.load(f)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        DATA_PATH, batch_size=1
    )

    print(f"\nDataLoader test:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # print the first batch
    for batch in train_loader:
        coorf, seg_matrix, p, sim_id = batch
        print(coorf.shape)
        print(seg_matrix.shape)
        print(p.shape)
        print(sim_id)
        break
