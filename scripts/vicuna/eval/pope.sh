# #!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

coco_dir=/mnt/workspace/xue.w/yf/data/coco
model_name="$1"
model_dir=/mnt/workspace/xue.w/yf/checkpoint/$model_name

for POPE_CAT in llava_pope_test; do
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    name=$checkpoint
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                --model-path $model_dir/$name \
                --question-file ./playground/data/eval/pope/${POPE_CAT}.jsonl \
                --image-folder $coco_dir/val2014 \
                --answers-file  ./playground/data/eval/pope/answers/$model_name-$name-${CHUNKS}_${IDX}.jsonl\
                --temperature 0.0 \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --conv-mode vicuna_v1 \
                --max_new_tokens 64 & #>> ./playground/$model_name-$name-${POPE_CAT}.log 2>&1 & 
    done
    
    wait
    output_file=./playground/data/eval/pope/answers/$model_name-$name-${POPE_CAT}.jsonl
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/pope/answers/$model_name-$name-${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done


    python llava/eval/eval_pope.py \
        --annotation-dir ./playground/data/eval/pope/coco \
        --question-file ./playground/data/eval/pope/${POPE_CAT}.jsonl \
        --result-file ./playground/data/eval/pope/answers/$model_name-$name-${POPE_CAT}.jsonl
done
    
done
