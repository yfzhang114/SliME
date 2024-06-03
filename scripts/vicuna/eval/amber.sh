#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo $CHUNKS
name="$1"
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    model_name=$name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_amber \
            --model-path $model \
            --question-file ./playground/data/eval/AMBER/data/query/query_discriminative.json \
            --image-folder ./playground/data/eval/AMBER/image \
            --answers-file ./playground/data/eval/AMBER/answers/$model_name-d${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0. \
            --conv-mode vicuna_v1 & # --use-qlora True --qlora-path $lora_path
    done
    wait

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_amber \
        --model-path $model \
        --question-file ./playground/data/eval/AMBER/data/query/query_generative.json \
        --image-folder ./playground/data/eval/AMBER/image \
        --answers-file ./playground/data/eval/AMBER/answers/$model_name-g${CHUNKS}_${IDX}.jsonl\
        --temperature 0. \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 & # --use-qlora True --qlora-path $lora_path
    done
    wait

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/AMBER/answers/$model_name-g${CHUNKS}_${IDX}.jsonl >> "./playground/data/eval/AMBER/answers/$model_name-g.jsonl"
        cat ./playground/data/eval/AMBER/answers/$model_name-d${CHUNKS}_${IDX}.jsonl >> "./playground/data/eval/AMBER/answers/$model_name-d.jsonl"
    done

    echo $model_name
    cd ./playground/data/eval/AMBER

    python inference.py --inference_data ./answers/$model_name-d.jsonl --evaluation_type d
    python inference.py --inference_data ./answers/$model_name-g.jsonl --evaluation_type g

done