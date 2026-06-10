#!/bin/bash
#SBATCH --job-name=train_domino                   
#SBATCH --cpus-per-task=1                     
#SBATCH --mem-per-cpu=250G                     
#SBATCH --gres=gpu:1                          

######################################################################  
# Example script for running Domino on Drosophila melanogaster data.
# Please customize all folder/file names and parameters for 
# your own experiment and dataset.
######################################################################  

# FOLDER NAMES
genome_path=dm6.fa
ins_dir="insulations/insulation_files"
bound_dir="boundaries"
out_folder="output"

# FILE NAMES
ins1_path='insulations/insulation_files/dm6_rep1.bed'
ins2_path='insulations/insulation_files/dm6_rep2.bed'
bound_path='boundaries/dm6_rep1.bed'
# if applicable
blacklist_path='annotations/dm6-blacklist.v2.bed'

# PARAMETERS
delta=2
ins_ws=1
demean_context_bin_width=10
DNA_context_bin_width=16
kernel_length=5
conv_layers=5
resolution=200
activation='gelu'

train_coords_dict='{"full_chr_names": ["chr2L", "chr3L", "chr3R", "chr4", "chrX", "chrY"]}'
val_coords_dict='{"chrom_to_subset": "chr2R", "min_start": 15000000}'
test_coords_dict='{"chrom_to_subset": "chr2R", "max_end": 14996800}'


python domino/training/train_domino_model.py --out_folder=$out_folder --genome_path=$genome_path \
                --delta=$delta --ins_window_size=$ins_ws --model_type='DominoSingleOutput' \
                --demean_context_bin_width=$demean_context_bin_width --DNA_context_bin_width=$DNA_context_bin_width \
                --ins1_path=$ins1_path --ins2_path=$ins2_path --bound_path=$bound_path \
                --train_coords_dict="${train_coords_dict}" --val_coords_dict="${val_coords_dict}" --test_coords_dict="${test_coords_dict}" \
                --boundary_length=200 --batch_size=32 --learning_rate=1e-4 --scheduler=plateau --conv_layers=$conv_layers \
                --conv_repeat=1 --kernel_number=512 --kernel_length=$kernel_length --filter_number=256 --filter_size=5 \
                --h_layers=4 --hidden_size=64 --activation=$activation 

echo "Finished training model."
