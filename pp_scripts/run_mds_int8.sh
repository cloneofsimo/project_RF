NUM_GPUS=8

START_INDEX=0
END_INDEX=11356

for ((i=START_INDEX; i<=END_INDEX; i++)); do

    GPU_INDEX=$((i % NUM_GPUS))

    CUDA_VISIBLE_DEVICES=$GPU_INDEX python vae_t5_preprocessing.py --device cuda --file_index $i &

    if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
        wait
    fi
done