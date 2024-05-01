#!/bin/bash
#SBATCH --job-name=detrend_normalize_features
#SBATCH --output=detrend_normalize_features%j.out
#SBATCH --error=detrend_normalize_features%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Mar4_feature_extraction

python feature_extraction.py
