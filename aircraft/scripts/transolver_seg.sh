#!/bin/bash
#SBATCH -J test_seg_v4
#SBATCH --output=./slurm_plus/test_seg_v4_%j.log
#SBATCH --error=./slurm_plus/test_seg_v4_%j.err
#SBATCH --account=bbqg-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4     # <- or one of: ghx4
#SBATCH --time=12:00:00      # hh:mm:ss for the jo
#SBATCH --mem=128g
# ### GPU options ###
#SBATCH --gpus-per-node=1

module load python/miniforge3_pytorch/2.5.0
which python3
echo "job is starting on `hostname`"
echo "SLURM_NODELIST=$SLURM_NODELIST"

python3 -m pip install --user --no-cache-dir \
    timm einops tqdm numpy scipy pandas h5py >/dev/null 2>&1

cd /u/wzhong/Aircraft_icml/

# Run with 2 GPUs using torchrun
# torchrun automatically sets up WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
python main_airplane.py --model_name Transolver_seg --nb_epochs 200 --fold_id 0 --dataset airplane


