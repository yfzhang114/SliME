#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="$1"

cd playground/data/eval/MMMU/eval

model=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint
model_name=$name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_llava.py \
        --config_path configs/llava1.5.yaml \
        --model_path $model \
        --answers-file ./answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --split "test" \
        --conv-mode vicuna_v1 & #--load_8bit True \ use this if you want to load 8-bit model
done

wait

output_file=./answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python convert_to_test.py --result_file $output_file --output_path ./$CKPT/test.json