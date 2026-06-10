#!/bin/bash

echo "Script started."

merged_input=nc14_embryo_merged.cool

# Zoomify the merged file
cooler zoomify -p 16 -r 100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000 --balance --out nc14_embryo_merged.mcool $merged_input

echo "Finished zoomifying the cool file."