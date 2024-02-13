#!/bin/bash

#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o play-%j.out  # %j = job ID
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gypsum-rtx8000
#SBATCH --mem=40G

python playground.py