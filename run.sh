#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu2019
#SBATCH --mem=85G
#SBATCH --mail-user=rylan.marianchuk@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --output="outfile.out"

python3 tm.py


