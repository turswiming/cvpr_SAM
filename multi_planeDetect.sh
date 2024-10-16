#!/bin/bash
Path="/media/lzq/Windows/Users/14318/scan2bim2024/3d/test/5cm"
N=5

# Get the list of files
files=($Path/*.ply)

# Run the Python script in parallel
for file in "${files[@]}"; do
    echo "Processing $file"
    /home/lzq/Desktop/denoise/.venv/bin/python planeDetect.py "$Path" "$(basename $file)" &
    # Limit the number of background jobs to prevent overloading the system
    while (( $(jobs | wc -l) >= N )); do
        sleep 1
    done
done

# Wait for all processes to finish
wait