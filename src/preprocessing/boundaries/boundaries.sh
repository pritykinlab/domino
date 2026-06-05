#!/bin/bash
#SBATCH --job-name=compute_boundaries                            # create a short name for your job
#SBATCH --nodes=1                                           # node count
#SBATCH --cpus-per-task=3                                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G                                    # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00                                      # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                                   # send email when job begins
#SBATCH --mail-type=end                                     # send email when job ends

echo "Script started."

resolution=200
window_size=48kb
insulation_fn='nc14_embryo_merged.insulation'
boundary_fn="nc14_embryo_merged.insulation_resolution_${resolution}_window_${window_size}_boundaries.bed"
fanc boundaries $insulation_fn $boundary_fn -w 48kb

echo "Script completed."