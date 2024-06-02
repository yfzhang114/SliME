#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name="$1"
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    
    model_name=$name-$checkpoint
    model=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint
    python -m llava.eval.model_vqa_loader \
        --model-path $model \
        --model-base meta-llama/Meta-Llama-3-8B-Instruct \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$model_name.jsonl \
        --temperature 0. \
        --conv-mode llama3 # --use-qlora True --qlora-path $lora_path

    echo $model_name
    cd ./playground/data/eval/MME

    python convert_answer_to_mme.py --experiment $model_name

    cd eval_tool

    python calculation.py --results_dir answers/$model_name

done