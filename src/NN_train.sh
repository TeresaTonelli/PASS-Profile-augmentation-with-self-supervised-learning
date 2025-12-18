#!/usr/bin/env bash

#SBATCH --time=01:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G  
#SBATCH --partition=boost_usr_prod

#SBATCH --output=make_src_outputs/make_train_1Unet.%j

#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --job-name=job_1Unet_make_train

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:1


eval "$(/leonardo/home/userexternal/ttonelli/miniconda3/bin/conda shell.bash hook)"
conda activate glob_io_env
#/leonardo_work/IscrC_MEDConNN/SSint_env

python NN_train.py \
    --total_epochs 50 \
    --save_every 25 \
    --batch_size 1 \
    --lr 0.0001 \
    --variable CHLA \
