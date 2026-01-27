import argparse
import os
import torch
from dataloader import ALL_JEB_DATA_PATH, load_data, discover_indices, create_data_loaders
from dataloader_seperate import DATA_PATH, create_data_loaders as create_data_loaders_seperate
from models.transolver_model import Model as Transolver_Model
from models.transolver_seg import Model as Transolver_seg_Model
from ml import train, test
import numpy as np


def build_index_splits(indices):
    """
    Build ~70/10/20 train/val/test splits using a simple round-robin scheme
    over the provided list of sample IDs (strings).
    """
    train_idx, val_idx, test_idx = [], [], []

    for i, idx in enumerate(indices):
        r = i % 10
        if r < 7:  # 0–6 → train
            train_idx.append(idx)
        elif r == 7:  # 7 → val
            val_idx.append(idx)
        else:  # 8–9 → test
            test_idx.append(idx)

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="transolver")
    parser.add_argument("--loader_type", type=str, default="seperate")
    parser.add_argument("--phase", type=str, default="train")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    model_name = args.model_name

    print(f"🚀 Starting JEB {model_name} Training")
    print(f"  Data path: {ALL_JEB_DATA_PATH}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n🔄 Creating split indexes...")
    from index import indexes
    # randomly select 500 indices from indexes
    all_indices = indexes # np.random.choice(indexes, size=500, replace=False)
    print(f"Using {len(all_indices)} JEB samples from index list")
    train_index, val_index, test_index = build_index_splits(all_indices)

    if args.loader_type == "seperate":
        print("\n🔄 Creating JEB data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders_seperate(
            data_folder=DATA_PATH,
            batch_size=1,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
        )
    elif args.loader_type == "consolidated":
        print("\n🔄 Loading JEB consolidated data...")
        data_dict = load_data(ALL_JEB_DATA_PATH)
        print(f"Loaded {len(data_dict)} samples from consolidated file")
        print("\n🔄 Creating JEB data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dict,  # Pass the loaded dict directly
            batch_size=1,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
            shuffle=True,
            num_workers=4,  # Added for performance
            pin_memory=True,  # Added for performance
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n🏗️  Creating Transolver model for JEB...")
    if model_name == "transolver":
        model = Transolver_Model(
            space_dim=3,   # volume_nodes (x, y, z)
            out_dim=1,     # scalar stress (or feature) per node
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
            space_dim=3,   # volume_nodes (x, y, z)
            out_dim=1,     # scalar stress (or feature) per node
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    save_path = "trained_models_JEB/best_{}_model.pth".format(model_name)
    # try to load the model if it exists
    if args.phase == "train" or args.phase == "test":
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"Loaded model from {save_path}")
        else:
            print(f"No model found at {save_path}")
    elif args.phase == "restart_train":
        pass
    
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n🎯 Starting JEB training...")
    history = train(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=args.learning_rate,
        eval_freq=args.eval_freq,
        save_path=save_path,
    )

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    print("\n🧪 Evaluating on train / val / test splits...")
    print("\n  Testing on training set...")
    _ = test(
        model=model,
        model_name=model_name,
        test_loader=train_loader,
        model_path=save_path,
    )

    print("\n  Testing on validation set...")
    _ = test(
        model=model,
        model_name=model_name,
        test_loader=val_loader,
        model_path=save_path,
    )

    print("\n  Testing on test set...")
    _ = test(
        model=model,
        model_name=model_name,
        test_loader=test_loader,
        model_path=save_path,
    )

    print("\n✅ JEB pipeline completed!")


if __name__ == "__main__":
    main()


