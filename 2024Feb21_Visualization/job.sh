#!/bin/bash
#SBATCH --job-name=visualization
#SBATCH --output=visualization%j.out
#SBATCH --error=visualization%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:20:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/2024Feb21_visualization

python visualization.py
