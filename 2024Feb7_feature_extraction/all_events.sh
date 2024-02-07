#!/bin/bash
#SBATCH --job-name=all_events_feature_extraction
#SBATCH --output=all_events_feature_extraction%j.out
#SBATCH --error=all_events_feature_extraction%j.err
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Feb7_feature_extraction

python all_events.py
