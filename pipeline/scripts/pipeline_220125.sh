#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 47:00:00
#SBATCH --exclusive
#SBATCH -n 10
#SBATCH -G 1

module load anaconda3
source activate preprocessing
 
logsave /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/test_run/output.log python /home/mpg08/aicha.hajiali/TLI_project/preprocessing/pipeline/scripts/general_pipeline_4D.py /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/test_run/general_pipeline_info.txt