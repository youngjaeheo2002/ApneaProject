#!/bin/bash
#SBATCH --job-name=ppg_sampling
#SBATCH --output=ppg_sampling_%j.out
#SBATCH --error=ppg_sampling_%j.err
#SBATCH --mem=128G
#SBATCH --time=2:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/muse_feature_extraction

python ppg_sampling.py
