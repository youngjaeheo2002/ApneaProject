#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=feature_extraction%j.out
#SBATCH --error=feature_extraction%j.err
#SBATCH --mem=128G
#SBATCH --time=9:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Feb12_Feature_Extraction

python ANNE_feature_extraction.py
