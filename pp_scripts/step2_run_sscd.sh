
wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt

NUM_GPUS=8
START_INDEX=0
END_INDEX=11356

export ROOT_DIR="/home/host/simo/capfusion_256"

for ((i=START_INDEX; i<=END_INDEX; i++)); do

    GPU_INDEX=$((i % NUM_GPUS))

    CUDA_VISIBLE_DEVICES=$GPU_INDEX python sscd_numpyemb.py --device cuda --file_index $i &

    if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
        wait
    fi
done