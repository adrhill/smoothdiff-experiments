#!/bin/bash
#SBATCH --job-name=sg100k
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb&amd"
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/job-%j.out
#SBATCH --mail-type=BEGIN,FAIL,END

echo -n $(hostname) ; cat /proc/cpuinfo | grep 'model name'|uniq | sed s/"model name"//g
apptainer run --nv apptainer.sif julia --project --color=yes run_100k.jl
