#!/bin/bash
# Job name:
#SBATCH --job-name=VizData
#
# Account:
#SBATCH --account=moabb
#
# Number of nodes:
#SBATCH --nodes=1
#
# Partition:
#SBATCH --partition=all
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=6
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:2
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-user=sebastien.velut@isae-supaero.fr
#SBATCH --mail-type=ALL
## Command(s) to run (example):
python ./Scripts/Visualize_data.py
