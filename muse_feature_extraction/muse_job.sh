#!/bin/bash
#SBATCH --job-name=ppg_extraction
#SBATCH --output=ppg_extraction_%j.out
#SBATCH --error=ppg_extraction_%j.err
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/muse_feature_extraction

python muse_visualization.py
