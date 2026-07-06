import train_airplane as train
import os
import torch
import argparse
from torch.utils.data import RandomSampler, DistributedSampler
import logging
from dataset.dataset import AirplaneDataLoader, AirplaneDataset
import torch.distributed as dist
import datetime
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/taiga/illinois/eng/cee/meidani/Vincent/aircraft_industry/processed_data3/')
parser.add_argument('--save_dir', default='/taiga/illinois/eng/cee/meidani/Vincent/aircraft_industry/processed_data3/')
parser.add_argument('--model_name', default='Transolver_plus', type=str)
parser.add_argument('--json_filename', default='/u/wzhong/Aircraft/airplane_dataset.json', type=str)
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
# add arguments related to normalization
parser.add_argument('--pos_norm', default=1, type=int)
parser.add_argument('--out_norm', default=1, type=int)
parser.add_argument('--dataset', default='drivernet')
parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--out-dim', default=4, type=int)
parser.add_argument('--world_size', type=int, default=None, 
                    help='Number of GPUs/processes for distributed training. If not set, uses WORLD_SIZE env var or defaults to 1 (single GPU)')
parser.add_argument('--master_addr', type=str, default=None,
                    help='Master address for distributed training. If not set, uses MASTER_ADDR env var or defaults to 127.0.0.1')
parser.add_argument('--master_port', type=str, default=None,
                    help='Master port for distributed training. If not set, uses MASTER_PORT env var or auto-generates')
args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

# Determine world size: use arg if provided, else env var, else default to 1
if args.world_size is not None:
    hosts = args.world_size
elif os.environ.get("WORLD_SIZE") is not None:
    hosts = int(os.environ.get("WORLD_SIZE"))
else:
    hosts = 1  # Single GPU default

rank = int(os.environ.get("RANK", "0"))  # process id
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
gpus = torch.cuda.device_count()  # gpus per node
args.local_rank = local_rank

# Only initialize distributed training if hosts > 1
if hosts > 1:
    # Get master address and port: use args if provided, else env vars, else defaults
    ip = args.master_addr if args.master_addr else os.environ.get("MASTER_ADDR", "127.0.0.1")
    if args.master_port:
        port = args.master_port
    elif os.environ.get("MASTER_PORT"):
        port = os.environ.get("MASTER_PORT")
    else:
        # Auto-generate a random port if not specified
        import random
        port = str(random.randint(10000, 65535))
    
    print(f"Initializing distributed training with {hosts} processes")
    print(f"MASTER_ADDR={ip}, MASTER_PORT={port}, RANK={rank}, LOCAL_RANK={local_rank}")
    
    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                            rank=rank, timeout=datetime.timedelta(seconds=100))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    # Single GPU mode - no distributed setup needed
    print(f"Running in single GPU mode on device 0")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(0)

train_dataset = AirplaneDataset(filename=args.json_filename, path=args.save_dir, train=True)
val_dataset = AirplaneDataset(filename=args.json_filename, path=args.save_dir, train=False)

# Use DistributedSampler for multi-node/multi-GPU training
if hosts > 1 and dist.is_initialized():
    train_sampler = DistributedSampler(train_dataset, num_replicas=hosts, rank=rank, shuffle=True, seed=0)
    val_sampler = DistributedSampler(val_dataset, num_replicas=hosts, rank=rank, shuffle=False, seed=0)
else:
    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(0))
    val_sampler = RandomSampler(val_dataset, generator=torch.Generator().manual_seed(0))

train_loader = AirplaneDataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
val_loader = AirplaneDataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

# 200 case
pos_mean = torch.tensor([2.80879162e+03, 1.00957077e+02, 6.76594237e-03]).view(1, 1, 3).cuda()
pos_std = torch.tensor([1436.65326859, 178.37956359, 615.16521715]).view(1, 1, 3).cuda()
norm_mean = torch.tensor([-7.03865828e-02, 1.50757955e-01, -6.07368549e-06]).view(1, 1, 3).cuda()
norm_std = torch.tensor([0.19895465, 0.87515866, 0.40665163]).view(1, 1, 3).cuda()
out_mean = torch.tensor([0.04602036, 1.3157164, 5.66693757, 0.25599, 0.06231503, 1.64027649]).view(1, 1, 6).cuda()
out_std = torch.tensor([0.09458788, 0.76978003, 0.41717544, 0.47068753, 0.6710297, 1.8059161]).view(1, 1, 6).cuda()

from models.Transolver_plus import Model as Transolver_plus
from models.Transolver_seg import Model as Transolver_seg
from torch.nn.parallel import DistributedDataParallel as DDP


if args.model_name == 'Transolver_plus':
    model = Transolver_plus(n_hidden=256, n_layers=4, space_dim=7,
                fun_dim=0,
                n_head=8,
                mlp_ratio=2, out_dim=6,
                slice_num=32,
                unified_pos=0,
                dropout=0.1).to(device)
elif args.model_name == 'Transolver_seg':
    model = Transolver_seg(n_hidden=256, n_layers=4, space_dim=7,
                fun_dim=0,
                n_head=8,
                mlp_ratio=2, out_dim=6,
                slice_num=32,
                unified_pos=0,
                dropout=0.1).to(device)
# Wrap model with DDP for gradient synchronization (only if multi-GPU)
if hosts > 1 and dist.is_initialized():
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
# default
# path = f'metrics/airplane/{args.cfd_model}/{args.dataset}/{args.fold_id}/{args.nb_epochs}_{args.weight}'

path = "./"

if not os.path.exists(path):
    os.makedirs(path)

if args.eval:
    logging.basicConfig(filename=os.path.join(path, 'test.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)
else:
    logging.basicConfig(filename=os.path.join(path, 'train.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)

logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
logging.info(model)
print(model)

try:
    if not args.eval:
        # train
        print("start model training")
        model = train.main(args.model_name, device, train_loader, val_loader, model, hparams, path, val_iter=args.val_iter, reg=args.weight, pos_norm=args.pos_norm, out_norm=args.out_norm, norm_norm=0, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=True)
        
        # Save final model to trained_models directory
        trained_models_dir = "./trained_models"
        os.makedirs(trained_models_dir, exist_ok=True)
        
        # Extract model from DDP wrapper if needed
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        
        final_model_path = os.path.join(trained_models_dir, f"best_{args.model_name}_model.pth")
        torch.save(model_to_save.state_dict(), final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")
    else:
        print("start model evaluation")
        data_dir_h5 = args.data_dir
        res_file = '/taiga/illinois/eng/cee/meidani/Vincent/aircraft_industry/result_new.csv'
        df = pd.read_csv(res_file)
        id = 0

        pos_mean = torch.tensor([2.80879162e+03, 1.00957077e+02, 6.76594237e-03]).view(1, 1, 3).cuda()
        pos_std = torch.tensor([1436.65326859, 178.37956359, 615.16521715]).view(1, 1, 3).cuda()
        norm_mean = torch.tensor([-7.03865828e-02, 1.50757955e-01, -6.07368549e-06]).view(1, 1, 3).cuda()
        norm_std = torch.tensor([0.19895465, 0.87515866, 0.40665163]).view(1, 1, 3).cuda()
        out_mean = torch.tensor([0.04602036, 1.3157164, 5.66693757, 0.25599, 0.06231503, 1.64027649]).view(1, 1, 6).cuda()
        out_std = torch.tensor([0.09458788, 0.76978003, 0.41717544, 0.47068753, 0.6710297, 1.8059161]).view(1, 1, 6).cuda()

        model = torch.load("./model_200.pth").cuda()
        l2re = 0
        for index, row in df.iloc[-14:].iterrows():
            idx = row['idx']
            Ma = row['Ma']
            alpha = row['alpha']
            beta = row['beta']
            in_file_h5 = os.path.join(data_dir_h5, f'{int(idx)}_{Ma}_{alpha}_{beta}.h5')

            with h5py.File(in_file_h5, 'r') as f:
                normals = f['normals'][:]
                pos = f['pos'][:]
                values = f['values'][:]
        
            with torch.no_grad():
                pos = torch.tensor(pos, dtype=torch.float32).view(1, -1, 3).cuda()
                normals = torch.tensor(normals, dtype=torch.float32).view(1, -1, 3).cuda()
                pos = (pos - pos_mean) / pos_std
                N = pos.shape[1]
                x = torch.cat([pos, torch.zeros((1, N, 1), dtype=torch.float32).cuda(), normals], dim=2)
                condition = torch.tensor([Ma, alpha, beta]).view(1, 3).cuda().float()
                out = model((x, pos, condition))
                out = out * out_std + out_mean
                out = out.cpu().numpy()
                l2re += np.linalg.norm(out[0, :, -1] - values[:, -1]) / np.linalg.norm(values[:, -1])
                # save output with name
                np.save(f"output/{idx}_{Ma}_{alpha}_{beta}.npy", out)
            id += 1

        print(f"Average L2RE: {l2re / id}")
finally:
    # Clean up distributed process group
    if hosts > 1 and dist.is_initialized():
        dist.destroy_process_group()
        
