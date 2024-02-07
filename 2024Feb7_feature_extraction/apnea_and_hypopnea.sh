#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=apnea_and_hypopnea_feature_extraction%j.out
#SBATCH --error=apnea_and_hypopnea_feature_extraction%j.err
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Feb7_feature_extraction

python apnea_and_hypopnea.py
