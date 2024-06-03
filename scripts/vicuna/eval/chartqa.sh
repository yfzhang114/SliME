#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name="$1"
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    model_name=$name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint
    python  llava/eval/model_vqa_chartqa.py \
        --model-path $model \
        --question-file ./playground/data/eval/ChartQA/ChartQA_Dataset/test/test.json \
        --image-folder ./playground/data/eval/ChartQA/ChartQA_Dataset/test/png \
        --answers-file ./playground/data/eval/ChartQA/answers/$model_name.jsonl \
        --temperature 0. \
        --conv-mode vicuna_v1

done