#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-12

module load cuda/12 cudnn

source ~/miniforge3/bin/activate GW_Bump
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/lensing/M01

srun python m01_72tE_mag.py
