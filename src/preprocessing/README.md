# Micro-C preprocessing

This module contains the scripts and code needed to preprocess raw micro-C reads into final contact maps that can be used to compute insulation scores and boundaries.

## Micro-C pipeline

The micro-C pipeline starts with raw fastq reads from your experiment and converts it to the final mcool file format. All the scripts needed are in the `microc` folder. 

1. Install Conda Environment

    Start by creating conda environment:

    ```
    conda env create --name envname --file=./microC_processing_env.yml
    ```

2. Pre-Alignment

    After setting up the environment, follow the pre-alignment [instructions from Micro-C documentation](https://micro-c.readthedocs.io/en/latest/pre_alignment.html). This step is crucial because it will be used to generate index and genome file for the downstream steps in the pipeline. 
 
3. Run Micro-C pipeline

    After this, you are now ready to run the pipeline. Add the following fields to `config.yml`. 
    - `samples`: this contains a list of all your samples. For instance, if your paired reads are `sample1_1.fastq.gz` and `sample1_2.fastq.gz`, you should add `sample1` as an item under samples. Then, if your fastq files are named different (by default, it is `{name of sample}_1.fastqz` and `{name of sample}_2.fastqz`), edit the `Snakefile` accordingly.
    - `genome_index`: path to genome index which is the output of bwa index. This is needed for bwa mem (alignment of reads).
    - `chrom_size_file`: path to file with chromosome sizes.

 Then, run `microc_pipeline.sh` which will output a cool file for each sample (2 fastq files).

4. [Optional] Merge

    If you need to merge cool files you generated, edit `merge_cooler.sh` to adapt to your file names.

5. [Optional] Zoomify

    If you need to zoomify cool files to generate `.mcool` files, edit `zoomify_coolers.sh` to adapt to your file names.

## Insulation scores

Scripts and code for insulation score pipeline is in `insulations` folder.

To compute insulation scores (output is an `.insulation` file), refer to and edit `insulation.sh`. Specify the window sizes for which you want to compute the insulation scores. By default, after generating the  `.insulation` file, this file will also generate the corresponding bed files for each window size. 

## Boundary calling

Scripts and code for insulation score pipeline is in `boundaries` folder.

To compute boundaries (output is a `.bed` file), you need a `.insulation` file.

If you need to customize computing insulation and/or calling TAD boundaries, feel free to see the [`fanc` documentation](https://vaquerizaslab.github.io/fanc/fanc.html) for insulation score and boundaries.

