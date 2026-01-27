from dataloader import create_data_loaders
from utils.metric import compute_relative_error
from models.transolver_model import Model as Transolver_Model
from models.Transolver_seg import Model as Transolver_SEG_Model
from models.Transolver_seg_v2 import Model as Transolver_SEG_V2_Model
from models.mlp import MLP as MLP_Model
# from models.figconv import FigConv_Model
# from models.figconv import Multi_grid_model

from utils.ml import train, test

import os
import torch
import argparse
import pickle
import numpy as np

# use argparse to get the model name and predicted feature name
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="transolver")
parser.add_argument('--predicted_feature_name', type=str, default="pressure")
parser.add_argument('--phase', type=str, default="train", choices=["train", "restart_train", "test"])
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=2e-5)
args = parser.parse_args()

# set the experiment settings
model_name = args.model_name
predicted_feature_name = args.predicted_feature_name
phase = args.phase
model_flag = model_name + '_' + predicted_feature_name
num_epochs = args.num_epochs

def main():
    """
    Main function to run training and testing.
    """
    print("🚀 Starting VTK Data Processing Pipeline")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")

    ALL_INDEX = [
        46, 48, 55, 66, 77, 140, 171, 211, 215, 216,
        242, 245, 252, 259, 298, 318, 323, 332, 354, 356,
        409, 416, 461, 474, 487, 517, 520, 551, 575, 584,
        631, 642, 652, 677, 708, 739, 754, 775, 780, 781,
        788, 794, 853, 854, 862, 865, 898, 910, 920, 991,
        1010, 1017, 1081, 1091, 1093, 1098, 1273, 1320, 1329, 1332,
        1342, 1377, 1421, 1426, 1430, 1433, 1441, 1443, 1455, 1459,
        1463, 1490, 1496, 1502, 1507, 1525, 1539, 1541, 1542, 1552,
        1562, 1565, 1662, 1664, 1695, 1702, 1711, 1713, 1716, 1757,
        1785, 1822, 1836, 1837, 1901, 1931, 1944, 1951, 1975, 2014,
        2049, 2065, 2070, 2079, 2088, 2117, 2125, 2129, 2155, 2178,
        2188, 2193, 2198, 2204, 2242, 2258, 2270, 2280, 2323, 2337,
        2338, 2380, 2404, 2462, 2501, 2526, 2532, 2536, 2548, 2557,
        2563, 2582, 2660, 2662, 2694, 2706, 2719, 2729, 2732, 2752,
        2783, 2788, 2791, 2867, 2870, 2875, 2900, 2952, 2966, 2975,
        3026, 3048, 3079, 3137, 3138, 3161, 3184, 3236, 3285, 3288,
        3291, 3326, 3328, 3342, 3349, 3389, 3428, 3451, 3458, 3473,
        3487, 3488, 3502, 3513, 3516, 3532, 3546, 3555, 3616, 3623,
        3637, 3650, 3658, 3671, 3680, 3704, 3719, 3757, 3790, 3801,
        3816, 3833, 3843, 3848, 3849, 3874, 3889, 3892, 3936, 3940,
    ]

    TRAIN_index, VAL_index, TEST_index = [], [], []

    for i, idx in enumerate(ALL_INDEX):
        r = i % 10
        if r < 7:        # 0–6 → 7/10 → train
            TRAIN_index.append(idx)
        elif r == 7:     # 7 → 1/10 → val
            VAL_index.append(idx)
        else:            # 8–9 → 2/10 → test
            TEST_index.append(idx)
    
    # '''
    # Debug
    # '''
    # TRAIN_index = [46]
    # VAL_index = [46]
    # TEST_index = [46]

    DATA_PATH = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/normalized/"
    with open(DATA_PATH + "normalization_scalars.pkl", "rb") as f:
        normalization_scalars = pickle.load(f)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        DATA_PATH, 
        batch_size=1, 
        train_index=TRAIN_index, 
        val_index=VAL_index, 
        test_index=TEST_index,
        shuffle=True,
        predicted_feature_name=predicted_feature_name
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    if model_name == 'transolver':
        # Improved hyperparameters for better performance
        model = Transolver_Model(
            space_dim=6, 
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2,        # Increased from 2 to 4
            slice_num=64
        )
    elif model_name == 'transolver_seg':
        model = Transolver_SEG_Model(
            space_dim=6, 
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2,        # Increased from 2 to 4
        )  # Changed from 3 to 6 for 6D features
    elif model_name == 'transolver_seg_v2':
        model = Transolver_SEG_V2_Model(
            space_dim=6, 
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # load the pre-trained model if it exists
    if phase != 'restart_train':
        try:
            if os.path.exists("trained_models/best_model_{}.pth".format(model_flag)):
                # weights_only=False is safe here since we're loading our own trained models
                model.load_state_dict(
                    torch.load(
                        "trained_models/best_model_{}.pth".format(model_flag),
                        map_location=torch.device('cpu'),
                        weights_only=False
                    )
                )
                print("Loaded pre-trained model")
        except:
            print("No compatible pre-trained model found")
    
    # Training
    if phase == 'train' or phase == 'restart_train':
        print("\n🎯 Starting training...")
        history = train(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalization_scalars=normalization_scalars,
            num_epochs=num_epochs,        # Increased epochs
            learning_rate=args.learning_rate,    # Increased learning rate for AdamW
            eval_freq=args.eval_freq,
            save_path="trained_models/best_model_{}.pth".format(model_flag),
            predicted_feature_name=predicted_feature_name
        )
    
    # Testing
    print("\n🧪 Starting testing...")
    print("\n Testing on training dataset...")
    _ = test(
        model_name=model_name,
        model=model,
        test_loader=train_loader,
        normalization_scalars=normalization_scalars,
        model_path="trained_models/best_model_{}.pth".format(model_flag),
        predicted_feature_name=predicted_feature_name
    )
    print("\n Testing on validation dataset...")
    _ = test(
        model_name=model_name,
        model=model,
        test_loader=val_loader,
        normalization_scalars=normalization_scalars,
        model_path="trained_models/best_model_{}.pth".format(model_flag),
        predicted_feature_name=predicted_feature_name
    )
    print("\n Testing on test dataset...")
    _ = test(
        model_name=model_name,
        model=model,
        test_loader=test_loader,
        normalization_scalars=normalization_scalars,
        model_path="trained_models/best_model_{}.pth".format(model_flag),
        predicted_feature_name=predicted_feature_name
    )
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main() 