#!/usr/bin/env bash
#SBATCH -c 20                              # Request one core
#SBATCH -t 1-04:05                         # Runtime in D-HH:MM format
#SBATCH -p medium                        # Partition to run in
#SBATCH --mem=250GB                        # Memory total in MiB (for all cores)
#SBATCH -o %j_sbatch.out  # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e %j_sbatch.err  # File to which STDERR will be written, including job ID (%j)

cd /n/data1/hms/dbmi/zaklab/mmd/ESGPTTaskQuerying/profiling/inovalon || exit

module load miniconda3/4.10.3

source activate ESGPTTaskQuerier2

PATH="/home/mbm47/.conda/envs/ESGPTTaskQuerier2/bin:$PATH" python run_profiling.py
