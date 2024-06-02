#!/bin/bash

model_name="$1"
# for checkpoint in checkpoint-148 checkpoint-444 checkpoint-740 checkpoint-1036 checkpoint-1480

for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    name=$model_name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$model_name/$checkpoint #liuhaotian/llava-v1.5-7b

    python -m llava.eval.model_vqa_science \
        --model-path $model \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode llama3

    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/$name.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/$name_output.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/$name_result.json

done