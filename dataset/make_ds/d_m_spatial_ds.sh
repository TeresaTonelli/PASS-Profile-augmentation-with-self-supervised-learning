#!/usr/bin/env bash

#SBATCH --time=00:19:30

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=4G  
#SBATCH --partition=lrd_all_serial

#SBATCH --output=make_ds_outputs/make_spatial_ds.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_make_spatial_ds

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate alkalinity_pysr

python download_and_make_spatial_ds.py