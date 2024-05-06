#!/bin/bash

#SBATCH --job-name=Final-MULTIGPU

#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=/home2/kanakala.ganesh/CLIP_PART_1/outputs/CLIP_reproduce.txt

cd ..

python run.py configs/standard/TEST_RUN.yaml
