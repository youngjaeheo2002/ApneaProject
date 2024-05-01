#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=feature_extraction%j.out
#SBATCH --error=feature_extraction%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=9:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Feb14_ANNE_feature_extraction

python ANNE_feature_extraction.py
