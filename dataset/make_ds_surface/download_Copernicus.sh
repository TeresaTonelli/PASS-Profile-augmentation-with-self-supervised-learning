#!/usr/bin/env bash

#SBATCH --time=00:00:20

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=16G  
#SBATCH --partition=lrd_all_serial

#SBATCH --output=make_ds_outputs/make_ds_surface.%j

#SBATCH --account=OGS23_PRACE_IT_1
#SBATCH --job-name=job_surf_ds

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate alkalinity_pysr

python download_Copernicus.py