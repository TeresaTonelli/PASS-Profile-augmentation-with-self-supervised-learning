#!/usr/bin/env bash

#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=6G  
#SBATCH --partition=dcgp_usr_prod

#SBATCH --output=make_src_outputs/boxplots_RMSE_1Unet.%j

#SBATCH --account=OGS23_PRACE_IT_1
#SBATCH --job-name=job_1Unet_boxplots

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate glob_io_env

python boxplots_showing.py