#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=deep_dating_cnn 

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/deep_dating_cnn1/bin/activate

srun python3 train.py
