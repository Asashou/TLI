#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 47:00:00
#SBATCH --exclusive
#SBATCH -n 10
#SBATCH -G 1
#SBATCH --mem-per-gpu=60G

module load cudnn
module load cuda/11.2.2
module load anaconda3
source activate preprocessing
 
logsave /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/2022/T4/220209/new_pipeline/neruon1/last/output.log python /home/mpg08/aicha.hajiali/TLI_project/preprocessing/pipeline/scripts/general_pipeline_4D.py /home/mpg08/aicha.hajiali/TLI_project/TLI_data/preprocessed/2022/T4/220209/new_pipeline/neruon1/last/general_pipeline_info.txt