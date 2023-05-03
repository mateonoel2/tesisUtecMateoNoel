#!/bin/bash
#SBATCH --job-name=dask_preprocess
#SBATCH --partition=investigacion
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --time=unlimited
#SBATCH --output=dask_preprocess.out
#SBATCH --error=dask_preprocess.err

# Load necessary modules
module load python/3.9.2

# Run Python script
python dask_preprocess.py