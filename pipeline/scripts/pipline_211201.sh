#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 47:00:00
#SBATCH --exclusive
#SBATCH -n 10
#SBATCH -G 1
#SBATCH --mem-per-gpu=60G

module load anaconda3
source activate preprocessing
 
logsave /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/211016/regi12/output.log python /home/mpg08/aicha.hajiali/TLI_project/preprocessing/pipline/scripts/general_pipline.py /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/211016/regi12/general_pipline_info.txt