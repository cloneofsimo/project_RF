#!/bin/bash
# Example code to schedule batch process job
# Path to the file to check
FILE_PATH="/home/host/simo/capfusion_256_mds/04000/data/shard.00000.mds"

# Number of GPUs
NUM_GPUS=8

# Index range
START_INDEX=10001
END_INDEX=11356

# Check for the existence of the file every 10 seconds
while true; do
    if [ -f "$FILE_PATH" ]; then
        echo "File found: $FILE_PATH"
        echo "Starting the job..."

        # Loop to process the files
        for ((i=START_INDEX; i<=END_INDEX; i++)); do
            GPU_INDEX=$((i % NUM_GPUS))
            CUDA_VISIBLE_DEVICES=$GPU_INDEX python vae_preprocessing.py --device cuda --file_index $i &

            # Wait for all processes to finish every NUM_GPUS iterations
            if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
                wait
            fi
        done

        echo "Job completed"
        break  # Exit the loop after processing
    else
        echo "File not found: $FILE_PATH. Checking again in 10 seconds..."
        sleep 10
    fi
done
