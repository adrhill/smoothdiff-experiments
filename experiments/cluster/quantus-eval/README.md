Build apptainer via
```bash
srun --partition=cpu-2h --pty bash
apptainer build apptainer.sif apptainer.def
```

Then submit job via
```bash
sbatch run.sh
```