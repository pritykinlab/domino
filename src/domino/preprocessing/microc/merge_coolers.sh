#!/bin/bash

echo "Script started."

# Define output file
output=nc14_embryo_merged.cool

# merging 2 files together
cool_files_to_merge=("nc14_embryo_1.cool" "nc14_embryo_2.cool")

# Merge all .cool files
cooler merge $output "${cool_files_to_merge[@]}" 

echo "Finished merging the cool files."

echo "Script completed successfully."
