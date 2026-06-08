# Domino

Domino, which stands for DNA-to-Insulation-Score Model for Analyzing Insulation of Chromatin, is a sequence-to-insulation deep learning framework. Here, we developed it specifically to model local chromatin insulation in Drosophila. In practice, however, it could be readily extended to any Hi-C or Micro-C (3C) datasets. This document outlines how to run Domino on your custom datasets.

## Package installation

To create a Conda environment for DOMINO, use the following command:
```
conda env create --name domino --file environment.yml
```

Install the domino package:
```
conda activate domino
pip install -e .
```

## Data preprocessing

After setting up the environment, you will want to convert raw data into contact maps. We provide scripts that you can reference in the scr/processing folder. Please refer to the README.md file in the folder for instructions.

Briefly, the folder contains a pipeline that assumes that your data comes in the raw FASTQ files (paired end reads) for 3C data. A Snakefile can be used to automate converting paired end reads in FASTQ form to a .cool file. From here, you can use merge and zoomify cool files to generate a .mcool file. The script also computes insulation score and boundaries using the FAN-C package.

## Model training and evaluation

From here, you can run python train_domino_model.py which has the following arguments:
- out_folder: name of folder to which you save your model
- delta: the delta parameter from FAN-C used to compute boundaries (default: 2)
- ins_window_size: window size for insulation score 
- model_type: type of model used to train; usually this will be DominoSingleOutput, which means to compute single insulation score
- demean_context_bin_width: number of bins (radius) used for mean-scale normalization
- DNA_context_bin_width: number of bins (radius) of DNA sequence used as input to the model 
- ins1_path: path to insulation score file corresponding to first dataset of replicates 
- ins2_path: path to insulation score file corresponding to second dataset of replicates (if it exists)
- train_coords_dict: dictionary of the form '{"full_chr_names": [list of chroms]}' used for training; example is '{"full_chr_names": ["chr2L", "chr3L", "chr3R", "chr4", "chrX", "chrY"]}'
- val_coords_dict: dictionary of the form '{"chrom_to_subset": chrom, "min_start": start_value}' used for validation; example is '{"chrom_to_subset": "chr2R", "min_start": 15000000}'
- test_coords_dict: dictionary of the form '{"chrom_to_subset": chrom, "max_end": max_value}' used for testing; example is '{"chrom_to_subset": "chr2R", "max_end": 14996800}'
- bound_path: path to boundary file corresponding to boundaries in first dataset of replicates
- boundary_length: size of each boundary; this will also be the resolution or bin size of the .mcool file used generate insulation scores and boundaries
- batch_size: number of data points used in each batch for training and evaluation
- learning_rate: learning rate (default: 1e-4)
- scheduler: learning rate scheduler (default: plateau)
- conv_layers: number of convolutional layers in model 
- conv_repeat: number of convolutional blocks in model
- kernel_number: number of kernels in first convolutional layer
- kernel_length: kernel size in first convolutional layer 
- filter_number: number of kernels in intermediate convolutional layers
- filter_size: kernel size in intermediate convolutional layers
- h_layers: number of hidden layers in head module 
- hidden_size: size of each hidden layer 
- activation: activation function (default: GeLU) 
- blacklist_path: path to list of blacklisted regions

After training, you will be able to visualize summary statistics of model performances in the folder. This includes scatterplots and histograms comparing predicted insulation with ground truth insulation.

Please note that there are two parameters (`ins1_path` and `ins2_path`) for insulation that the module takes as input. This is because we assume that you have two independent datasets, where one is used for training/evaluation/testing and the other for solely assessing replicate dataset performance. However, if you want only want to use one dataset and do not need to assess replicate dataset performance, you can set both `ins1_path` and `ins2_path` to be the same.
