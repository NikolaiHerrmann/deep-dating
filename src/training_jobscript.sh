#!/bin/bash

#SBATCH --time=30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=job-%j.log
#SBATCH --mem=12G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=deep_dating_cnn 

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-learn/0.24.1-foss-2022a
module load OpenCV/4.6.0-foss-2022a-contrib
module load scikit-image/0.19.3-foss-2022a

python3 vgg.py
