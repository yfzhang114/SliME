#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

# nohup  bash scripts/dpo/gqa.sh >> /usr/local/node/output.log 2>&1 &

name="$1"
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do


model_name=$name-$checkpoint
model_dir=/mnt/workspace/xue.w/yf/checkpoint/$name
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $model_dir/$checkpoint \
        --model-base meta-llama/Meta-Llama-3-8B-Instruct \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$model_name/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0. \
        --conv-mode llama3 &
        # --use-qlora True --qlora-path $lora_path &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$model_name/merge.jsonl
echo $output_file
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$model_name/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
done