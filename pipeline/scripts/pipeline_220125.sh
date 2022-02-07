#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 47:00:00
#SBATCH --exclusive
#SBATCH -n 10
#SBATCH -G 1

module load anaconda3
source activate preprocessing
 
logsave /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/2021/211104/steps/postshift/output.log python /home/mpg08/aicha.hajiali/TLI_project/preprocessing/pipeline/scripts/general_pipeline_4D.py /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/2021/211104/steps/postshift/general_pipeline_info_1.txt