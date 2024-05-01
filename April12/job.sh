#!/bin/bash
#SBATCH --job-name=non_sleep_removal
#SBATCH --output=j%j.out
#SBATCH --error=j%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/April12
python no_sleep_removal.py
