#!/bin/bash
#SBATCH --job-name=ANNE_CNN_data_and_targets
#SBATCH --output=ANNE_CNN_data_and_targets%j.out
#SBATCH --error=ANNE_CNN_data_and_targets%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/CNN_Approach

python ANNE_ppg_CNN_data_and_targets.py
