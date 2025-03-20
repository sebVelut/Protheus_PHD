#!/bin/bash
#SBATCH --job-name=job-array   # nom du job
#SBATCH --account=moabb
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=%x_%A_%a.out   # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-5%3         # 20 travaux en tout mais 5 travaux max dans la file
# Wall clock limit:
#SBATCH --time=48:00:00
#SBATCH --mail-user=sebastien.velut@isae-supaero.fr
#SBATCH --mail-type=ALL
 
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load ...
 
# echo des commandes lancées
set -x
 
# Execution du binaire "mon_exe" avec des donnees differentes pour chaque travail
# La valeur de ${SLURM_ARRAY_TASK_ID} est differente pour chaque travail.
srun ./STL.py --clf_name ${SLURM_ARRAY_TASK_ID}