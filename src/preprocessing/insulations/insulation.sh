#!/bin/bash
#SBATCH --job-name=compute_insulation_score                            # create a short name for your job
#SBATCH --nodes=1                                           # node count
#SBATCH --cpus-per-task=3                                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G                                    # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00                                      # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                                   # send email when job begins
#SBATCH --mail-type=end                                     # send email when job ends

echo "Script started."

rep1_input_fn=nc14_embryo_merged.mcool@200
rep1_output_fn=nc14_embryo_merged.insulation
fanc insulation $rep1_input_fn $rep1_output_fn -w 1kb 2kb 4kb 8kb 16kb 32kb 48kb 64kb 80kb

# generate bed files
fanc insulation nc14_embryo_merged.insulation \
                -o bed

echo "Finished computing insulation scores."