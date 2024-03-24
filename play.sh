#!/bin/bash
#SBATCH -p gpu-preempt		  # Partition
#SBATCH --constraint vram40
#SBATCH -G 1  # Number of GPUs
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=50GB  # Requested Memory
#SBATCH -t 0-12:00:00  # Zero day, one hour
#SBATCH -o submittask%j.out  # Specify where to save terminal output, %j = job ID will be filled by slurm

module load miniconda/22.11.1-1
conda activate 696hw1
python test_util.py