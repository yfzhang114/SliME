#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_name="$1"
# for checkpoint in checkpoint-148 checkpoint-444 checkpoint-740 checkpoint-1036 checkpoint-1480

for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    name=$model_name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$model_name/$checkpoint #liuhaotian/llava-v1.5-7b
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $model \
            --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder /mnt/workspace/xue.w/yf/data/textvqa/train_images \
            --answers-file ./playground/data/eval/textvqa/answers/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    output_file=./playground/data/eval/textvqa/answers/$name.jsonl
    echo $output_file
    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/textvqa/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python -m llava.eval.eval_textvqa \
        --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
        --result-file ./playground/data/eval/textvqa/answers/$name.jsonl

done