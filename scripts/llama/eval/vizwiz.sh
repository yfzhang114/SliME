#!/bin/bash

model_name="$1"

python -m llava.eval.model_vqa_loader \
    --model-path  /mnt/data/xue.w/yf/checkpoint/$model_name \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode llama3

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$model_name.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$model_name.json
