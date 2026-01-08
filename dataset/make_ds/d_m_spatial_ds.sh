#!/usr/bin/env bash

#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=2G  
#SBATCH --partition=lrd_all_serial

#SBATCH --output=make_ds_outputs/make_spatial_ds.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_make_spatial_ds

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate alk_pysr_update

python download_and_make_spatial_ds.py