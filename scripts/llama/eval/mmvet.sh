#!/bin/bash


model=${1:-"liuhaotian/llava-v1.5-7b"}
name=$model

python -m llava.eval.model_vqa \
    --model-path /mnt/data/xue.w/yf/checkpoint/$model \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$model.jsonl \
    --temperature 0 \
    --conv-mode llama3

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$model.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$model.json

python ./playground/data/eval/mm-vet/evaluator.py \
    --result_file ./playground/data/eval/mm-vet/results/$model.json