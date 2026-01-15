#!/usr/bin/env bash

#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G  
#SBATCH --partition=boost_usr_prod

#SBATCH --output=make_src_outputs/boxplots_RMSE_1Unet.%j

#SBATCH --account=IscrC_MEDConNN 
#SBATCH --job-name=job_1Unet_boxplots

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate glob_io_env

python boxplots_showing.py