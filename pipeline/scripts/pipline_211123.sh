#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 47:00:00
#SBATCH --exclusive
#SBATCH -n 10
#SBATCH -G 1
#SBATCH --mem-per-gpu=60G

module load anaconda3
source activate preprocessing
 
logsave /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/211113/3steps/output.log python /home/mpg08/aicha.hajiali/TLI_project/preprocessing/pipline/scripts/general_preprocessing_pipline_211123.py /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/211113/3steps/general_pipline_info.txt