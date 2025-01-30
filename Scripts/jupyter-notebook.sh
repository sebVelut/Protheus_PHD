#!/bin/bash

# Run an interactive session launching jupyter notebook

#OPTIONS="$OPTIONS --account=verbockhaven"
#SBATCH --job-name=jupyter-notebook
#OPTIONS="$OPTIONS -p besteffort"                # best effort

#OPTIONS="$OPTIONS --output=jupyter.log"         # Log outpu
#OPTIONS="$OPTIONS --nodelist titanic-5"         # list of node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # number of CPU requested
#SBATCH --gres=gpu:1	             # number of GPU requested
#SBATCH -t 08:00:00                  # max runtime is 9 hours

echo "srun $OPTIONS node_jupyter.sh" 
srun $OPTIONS node_jupyter.sh

