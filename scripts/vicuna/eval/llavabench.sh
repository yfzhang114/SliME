#!/bin/bash

model_name="$1"
# for checkpoint in checkpoint-148 checkpoint-444 checkpoint-740 checkpoint-1036 checkpoint-1480

for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    name=$model_name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$model_name/$checkpoint
    
    python -m llava.eval.model_vqa \
        --model-path $model \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

    python llava/eval/eval_gpt_review_bench.py \
        --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
        --output \
            playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

    python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

done