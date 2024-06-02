#!/bin/bash

model_name="$1"
# for checkpoint in checkpoint-148 checkpoint-444 checkpoint-740 checkpoint-1036 checkpoint-1480
# model_name=LLaVA-13b-13b_vl_faithful-withllavarlhf-mpo-beta1-lambda1
for checkpoint in checkpoint-2000 '' #checkpoint-3000 checkpoint-6000 
do
    name=$model_name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$model_name/$checkpoint
    
    python -m llava.eval.model_vqa_mmhal \
    --model-path $model \
    --temperature 0.0 \
    --answers-file \
    ./eval/mmhal/answer-file-${name}.json --image_aspect_ratio pad --test-prompt ''

    # python llava/eval/eval_gpt_mmhal.py \
    #     --response ./eval/answer-file-${name}.json \
    #     --evaluation ./eval/review-file-${name}.json \
    #     --api-key "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" \
    #     --gpt-model gpt-4-0613

    # python llava/eval/summarize_gpt_mmhal.py \
    #     --evaluation ./eval/review-file-${MODEL_SUFFIX}.json

done