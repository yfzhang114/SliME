# #!/bin/bash
coco_dir=/mnt/workspace/xue.w/yf/data/coco
root_path=/mnt/data/xue.w/yf/checkpoint/
model_base=$root_path/LLaVA-RLHF-13b-v1.5-336/llava_sft_model
model_path=$root_path/LLaVA-RLHF-13b-v1.5-336/llava_rlhf_lora_adapter_model
for POPE_CAT in adversarial popular random; do
    nohup python -m llava.eval.model_vqa_loader \
        --model-base $model_base \
        --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl \
        --image-folder $coco_dir/val2014 \
        --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b_${POPE_CAT}.jsonl \
        --temperature 0.2 --max_new_tokens 64 \
        --test-prompt '\nAnswer the question using a single word or phrase.' \
        --conv-mode vicuna_v1 >> ./playground/llava-v1.5-7b-${POPE_CAT}.log 2>&1 & 

    python llava/eval/eval_pope.py \
        --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl \
        --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b_${POPE_CAT}.jsonl 
done

# coco_dir=/mnt/workspace/xue.w/yf/data/coco
# model_dir=/mnt/data/xue.w/yf/checkpoint/LLaVA-dpo-13b-v1.5-336-only-sft
# for POPE_CAT in adversarial; do
# for checkpoint in checkpoint-45 checkpoint-90 checkpoint-180 checkpoint-270 checkpoint-360 checkpoint-450
# do
#     name=$checkpoint
# #  nohup python -m llava.eval.model_vqa_loader \
# #             --model-path $model_dir/$name \
# #             --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl \
# #             --image-folder $coco_dir/val2014 \
# #             --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b-sft-$name-${POPE_CAT}.jsonl \
# #             --temperature 0.2 \
# #             --conv-mode vicuna_v1 \
# #             --test-prompt '\nAnswer the question using a single word or phrase.' \
# #             --max_new_tokens 64  >> ./playground/llava-v1.5-13b-sft-$name-${POPE_CAT}.log 2>&1 & 

#     python llava/eval/eval_pope.py \
#         --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl \
#         --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b-sft-$name-${POPE_CAT}.jsonl
# done
    
# done
# #!/bin/bash

# POPE Evaluation
# HF_HOME=/mnt/data/xue.w/yf/checkpoint
# export CUDA_VISIBLE_DEVICES=6
# MODEL_BASE=liuhaotian/llava-v1.5-7b  #LLaVA-RLHF-7b-v1.5-224/sft_model liuhaotian/llava-v1.5-13b

# MODEL_QLORA_BASE=~/yifan-germany/models/LLaVA-dpo-7b-v1.5-224-lora
# echo $MODEL_QLORA_BASE
# name=llava-v1.5-7b-rlhf
# # 遍历所有子文件夹
# for folder in "$MODEL_QLORA_BASE"/*; do
#     if [ -d "$folder" ]; then
#         mv "$folder/lora_default/"* "$folder/"
#         folder_name=$name-$(basename "$folder")
#         echo $folder_name
#         for POPE_CAT in adversarial; do
#             echo ${MODEL_QLORA_BASE} ${POPE_CAT} 
#             python -m llava.eval.model_vqa_loader  \
#             --model-path ${MODEL_BASE} \
#             --qlora-path ${folder} \
#                 --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl \
#                 --image-folder  $coco_dir/val2014 \
#                 --answers-file \
#                 ./playground/data/eval/pope/answers/${folder_name}_${POPE_CAT}.jsonl 
#             python llava/eval/eval_pope.py  \
#                 --result-file ./playground/data/eval/pope/answers/${folder_name}_${POPE_CAT}.jsonl \
#                 --question-file ./playground/data/eval/pope/coco_pope_${POPE_CAT}.jsonl 
# done
#     fi
# done
