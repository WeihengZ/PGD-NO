import argparse
import os

import torch

from dataloader import DATAPATH, load_data, create_data_loaders
from models.transolver_model import Model as Transolver_Model
from models.transolver_seg import Model as Transolver_Seg_Model
from ml import train, test


def build_index_splits(num_samples: int):
    """
    Build ~70/10/20 train/val/test splits using a simple round-robin scheme.
    """
    all_indices = list(range(num_samples))
    train_idx, val_idx, test_idx = [], [], []

    for i, idx in enumerate(all_indices):
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
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="transolver")
    parser.add_argument("--phase", type=str, default="train")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    model_name = args.model_name
    phase = args.phase
    print(f"🚀 Starting Heatsink {model_name} Training")
    print(f"  Data path: {DATAPATH}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n🔄 Creating heatsink data loaders...")
    data_dict = load_data(DATAPATH)
    num_samples = len(data_dict["vertices"])
    print(f"Found {num_samples} heatsink samples.")

    train_index, val_index, test_index = build_index_splits(num_samples)

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dict,
        batch_size=1,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        shuffle=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n🏗️  Creating {model_name} model for heatsink...")
    if model_name == "transolver":
            model = Transolver_Model(
            space_dim=3,   # vertices (x, y, z)
            out_dim=1,     # scalar temperature per vertex
            n_layers=8,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=2,
        slice_num=64,
        )
    elif model_name == "transolver_seg":
        model = Transolver_Seg_Model(
            space_dim=3,   # vertices (x, y, z)
            out_dim=1,     # scalar temperature per vertex
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

    save_path = "trained_models_heatsink/best_{}_model.pth".format(model_name)
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
    print("\n🎯 Starting heatsink training...")
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

    print("\n✅ Heatsink pipeline completed!")


if __name__ == "__main__":
    main()


