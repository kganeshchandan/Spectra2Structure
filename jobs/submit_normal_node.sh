#!/bin/bash

#SBATCH --job-name=Final-MULTIGPU
#SBATCH -A d4
#SBATCH -p ihub
#SBATCH -c 40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=/home2/kanakala.ganesh/CLIP_PART_1/outputs/OG_UNITNORM.txt



cd ..

python run.py configs/standard/unit_norm_proper_decoder.yaml
