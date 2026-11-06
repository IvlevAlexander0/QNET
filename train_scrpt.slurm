#!/bin/bash
#SBATCH --partition=p5gpuA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=3-00:00:00
#SBATCH --output=train_%j.out

python classifier_train.py