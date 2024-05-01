#!/bin/bash
#SBATCH --job-name=ANNE_CNN_model
#SBATCH --output=ANNE_CNN_data_model%j.out
#SBATCH --error=ANNE_CNN_model%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=168:00:00

source /home/heoyoun1/ENV/bin/activate
cd /home/heoyoun1/projects/def-alim/heoyoun1/ApneaProject/CNN_Approach

python ANNE_CNN_Model.py