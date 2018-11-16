#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:k80:4
#SBATCH -t 01:00:00
#SBATCH --verbose

#SBATCH --mail-user=khan74@illinois.edu
#SBATCH --mail-type=BEGIN

#SBATCH --job-name="high_prob-crossmatch"
#SBATCH -C EGRESS

#echo commands to stdout
set -x

#run GPU program
cd /home/khan74/scratch/new_DL_DES/

python /home/khan74/scratch/new_DL_DES/src/des/des_sdss_overlap/high_prob_overlap/scripts/cross_match.py
