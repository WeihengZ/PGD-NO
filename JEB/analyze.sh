#!/bin/bash
#SBATCH -J jeb_analyze
#SBATCH --output=./slurm_icml/jeb_analyze_%j.log
#SBATCH --error=./slurm_icml/jeb_analyze_%j.err
#SBATCH --account=bbqg-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64    # <- match to OMP_NUM_THREADS
#SBATCH --partition=ghx4     # <- or one of: ghx4 gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=12:00:00      # hh:mm:ss for the jo
#SBATCH --mem=512g
# ### GPU options ###
#SBATCH --gpus-per-node=1
# Note: --gpus-per-task is not needed when using torchrun (it manages processes internally)

module load python/miniforge3_pytorch/2.5.0
which python3
echo "job is starting on `hostname`"
echo "SLURM_NODELIST=$SLURM_NODELIST"

cd /u/wzhong/JEB/

python analyze.py