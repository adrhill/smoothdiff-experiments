#!/bin/bash
#SBATCH --job-name=sd_quantus
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb&cuda129"
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/job-%j.out
#SBATCH --mail-type=BEGIN,FAIL,END

echo -n $(hostname) ; cat /proc/cpuinfo | grep 'model name'|uniq | sed s/"model name"//g
echo "Running VGG19..."
apptainer run --nv apptainer.sif python run_vgg19.py
echo "Running ResNet18..."
apptainer run --nv apptainer.sif python run_resnet18.py
