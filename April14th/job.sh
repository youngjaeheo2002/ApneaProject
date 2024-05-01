#!/bin/bash
#SBATCH --job-name=sleep_period_feature_extraction
#SBATCH --output=j%j.out
#SBATCH --error=j%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/April14th
python sleep_periods_event_feature_extraction.py 60 10
