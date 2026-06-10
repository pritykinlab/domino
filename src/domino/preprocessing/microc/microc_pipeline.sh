#!/bin/bash
#SBATCH --job-name=microc_pipeline  
#SBATCH --nodes=1
#SBATCH --time=80:00:0
#SBATCH --mem=100000
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=begin                     
#SBATCH --mail-type=end                       

# converts raw fastq files into output.pairs files

# Run Snakemake workflow
snakemake -j 18 --keep-going \
    --jobs 10 --cores 16 \
    --latency-wait 120 all

echo "Snakemake workflow completed."

# Index pairix files
for f in *_output.pairs.gz; do
  pairix -f "${f}"
done

echo "Pairix indexing completed."

# Convert pairix files to cooler format
for f in *output.pairs.gz; do 
	STEM=$(basename "${f}" _output.pairs.gz)
	cooler cload pairix dm6.chrom.sizes:100 "${f}" "${STEM}".cool
done

echo "Pairix to cooler conversion completed."

# merge files afterwards if needed (see merge_coolers.sh)

echo "Script execution completed."

