#!/bin/bash
#SBATCH --job-name=pxflip
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb&amd"
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/job-%j.out
#SBATCH --mail-type=BEGIN,FAIL,END

echo -n $(hostname) ; cat /proc/cpuinfo | grep 'model name'|uniq | sed s/"model name"//g
apptainer run --nv apptainer.sif julia --project --color=yes run.jl
