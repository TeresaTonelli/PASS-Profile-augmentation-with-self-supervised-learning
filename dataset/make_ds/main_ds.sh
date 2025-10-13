#!/usr/bin/env bash

#SBATCH --time=00:00:30

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=128G  
#SBATCH --partition=boost_usr_prod

#SBATCH --output=make_ds.%j

#SBATCH --account=IscrC_MEDConNN
#SBATCH --job-name=job_make_ds

#SBATCH --cpus-per-task=1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate glob_io_env

python main_ds.py \
    --training_folder STint_FLOAT \
    --input_float_directory /leonardo_scratch/large/userexternal/ttonelli/OGS/STint_FLOAT/SUPERFLOAT \
    --flag_toy True \
    --variable CHLA \